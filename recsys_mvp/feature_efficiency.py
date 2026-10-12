from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import catboost
import duckdb
import numpy as np
import pandas as pd

from recsys_mvp.auto_features import (
    FeatureSpec,
    build_generated_feature_table,
    generate_feature_specs,
)
from recsys_mvp.catboost_tuning import _manual_feature_specs
from recsys_mvp.data import (
    DataContractError,
    create_examples_view,
    quote_sql_string,
    register_source,
)
from recsys_mvp.feature_search import CATEGORICAL_FEATURES, MANUAL_FEATURE_NAMES, _fit_and_score
from recsys_mvp.evaluation import evaluate_predictions
from recsys_mvp.sampling_sensitivity import _query_ndcg_values, _query_recall_values
from recsys_mvp.yambda import prepare_yambda_candidates


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _progress(stage: str, **values: Any) -> None:
    print(json.dumps({"stage": stage, **values}, ensure_ascii=False), flush=True)


def _connection() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.execute("SET memory_limit = '6GB'")
    return connection


def _examples(
    connection: duckdb.DuckDBPyConnection, path: Path,
    query_ids: pd.DataFrame | None = None,
) -> None:
    columns = register_source(connection, path)
    if query_ids is not None:
        connection.register("batch_queries", query_ids)
        connection.execute("CREATE TEMP TABLE batch_source AS SELECT source.* FROM source_data AS source JOIN batch_queries USING (query_id)")
        connection.execute("DROP VIEW source_data")
        connection.execute("CREATE TEMP VIEW source_data AS SELECT * FROM batch_source")
    create_examples_view(
        connection, columns, "user_id", "item_id", "timestamp", "label", "query_id"
    )


def _load_specs(manifest: dict[str, Any], variant: str) -> tuple[FeatureSpec, ...]:
    return tuple(
        FeatureSpec(**{key: value for key, value in row.items() if key != "name"})
        for row in manifest["feature_specs"][variant]
    )


def _peak_rss_mb() -> float:
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def _benchmark_worker(manifest_path: Path, variant: str, output: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    specs = _load_specs(manifest, variant)
    connection = _connection()
    try:
        setup_started = time.perf_counter()
        _examples(connection, Path(manifest["audit_candidates"]))
        setup_seconds = time.perf_counter() - setup_started
        started = time.perf_counter()
        build_generated_feature_table(connection, Path(manifest["event_history"]), specs)
        materialized_seconds = time.perf_counter() - started
        rows = connection.execute("SELECT * FROM generated_model_data").fetchdf()
        elapsed = time.perf_counter() - started
        _write_json(output, {
            "variant": variant,
            "setup_seconds": setup_seconds,
            "materialize_seconds": materialized_seconds,
            "build_and_fetch_seconds": elapsed,
            "peak_rss_mb": _peak_rss_mb(),
            "rows": len(rows),
            "numeric_matrix_bytes": int(rows[[spec.name for spec in specs]].memory_usage(index=False).sum()),
            "feature_sums": {spec.name: float(rows[spec.name].fillna(0).sum()) for spec in specs},
        })
    finally:
        connection.close()


def _cluster_intervals(
    frame: pd.DataFrame, manual_scores: np.ndarray, automatic_scores: np.ndarray,
    resamples: int = 5000,
) -> dict[str, Any]:
    manual = _query_ndcg_values(frame, manual_scores)
    automatic = _query_ndcg_values(frame, automatic_scores)
    manual_recall = _query_recall_values(frame, manual_scores)
    automatic_recall = _query_recall_values(frame, automatic_scores)
    if manual.keys() != automatic.keys() or not manual:
        raise DataContractError("Paired query sets differ or are empty.")
    values = pd.DataFrame([
        {"user_id": user, "queries": 1,
         "ndcg_delta": automatic[(user, query)] - manual[(user, query)],
         "noninferiority_margin": automatic[(user, query)] - 0.98 * manual[(user, query)],
         "recall_delta": automatic_recall[(user, query)] - manual_recall[(user, query)]}
        for user, query in sorted(manual)
    ])
    columns = ["ndcg_delta", "noninferiority_margin", "recall_delta"]
    clusters = values.groupby("user_id", sort=True)[["queries", *columns]].sum().to_numpy()
    rng = np.random.default_rng(42)
    bootstraps = []
    for start in range(0, resamples, 64):
        indices = rng.integers(0, len(clusters), size=(min(64, resamples - start), len(clusters)))
        totals = clusters[indices].sum(axis=1)
        bootstraps.append(totals[:, 1:] / totals[:, :1])
    intervals = np.quantile(np.concatenate(bootstraps), [0.025, 0.975], axis=0)
    return {
        "paired_queries": len(values), "paired_users": len(clusters),
        "resamples": resamples, "seed": 42, "bootstrap_unit": "user",
        **{column: {"estimate": float(values[column].mean()),
                    "bootstrap_95_percent_interval": intervals[:, index].tolist()}
           for index, column in enumerate(columns)},
    }


def _quality(manifest: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    connection = _connection()
    specs = {spec.name: spec for variant in ("manual", "automatic")
             for spec in _load_specs(manifest, variant)}
    try:
        _examples(connection, Path(manifest["candidates"]))
        cutoffs = connection.execute("""
            SELECT quantile_disc(timestamp, 0.50), quantile_disc(timestamp, 0.85), MAX(timestamp)
            FROM (SELECT DISTINCT query_id, timestamp FROM source_data)
        """).fetchone()
        if any(value is None for value in cutoffs):
            raise DataContractError("The experiment has no temporal cutoffs.")
        query_ids = connection.execute(
            "SELECT DISTINCT query_id, timestamp FROM source_data ORDER BY timestamp, query_id"
        ).fetchdf()[["query_id"]]
        source_sql = quote_sql_string(manifest["candidates"])
        audit_sql = quote_sql_string(manifest["audit_candidates"])
        connection.execute(f"""
            COPY (SELECT * FROM read_parquet({source_sql}) WHERE timestamp > ?)
            TO {audit_sql} (FORMAT PARQUET, COMPRESSION ZSTD)
        """, [cutoffs[1]])
        audit_queries, audit_users, audit_positives = connection.execute(f"""
            SELECT COUNT(DISTINCT query_id), COUNT(DISTINCT user_id), SUM(label)
            FROM read_parquet({audit_sql})
        """).fetchone()
        _progress("quality_features", audit_queries=audit_queries, audit_users=audit_users)
    finally:
        connection.close()
    matrices = []
    started = time.perf_counter()
    for offset in range(0, len(query_ids), 2000):
        connection = _connection()
        try:
            _examples(connection, Path(manifest["candidates"]), query_ids.iloc[offset:offset + 2000])
            build_generated_feature_table(connection, Path(manifest["event_history"]), tuple(specs.values()))
            matrices.append(connection.execute("SELECT * FROM generated_model_data ORDER BY row_id").fetchdf())
        finally:
            connection.close()
        _progress("quality_feature_batch", completed_queries=min(offset + 2000, len(query_ids)), total_queries=len(query_ids))
    rows = pd.concat(matrices, ignore_index=True)
    del matrices
    feature_seconds = time.perf_counter() - started
    for column in CATEGORICAL_FEATURES:
        rows[column] = rows[column].fillna("__MISSING__").astype(str)
    train = rows["event_time"] <= cutoffs[0]
    selection = (rows["event_time"] > cutoffs[0]) & (rows["event_time"] <= cutoffs[1])
    audit = rows["event_time"] > cutoffs[1]
    if not train.any() or not selection.any() or not audit.any():
        raise DataContractError("The experiment has an empty partition.")
    results, scores = {}, {}
    audit_rows = rows.loc[audit]
    for variant in ("manual", "automatic"):
        names = tuple(spec.name for spec in _load_specs(manifest, variant))
        _progress("fit_selection", variant=variant, rows=int(train.sum()))
        selection_model, selection_result = _fit_and_score(rows, names, train, selection)
        del selection_model
        _progress("fit_audit", variant=variant, iterations=selection_result["iterations"])
        model, result = _fit_and_score(rows, names, train | selection, audit, selection_result["iterations"])
        model_path = output_dir / f"{variant}.cbm"
        model.save_model(str(model_path))
        scores[variant] = model.predict_proba(audit_rows[[*CATEGORICAL_FEATURES, *names]])[:, 1]
        results[variant] = {"selection": selection_result, "audit": result,
                            "model_bytes": model_path.stat().st_size,
                            "audit_feature_sums": {name: float(audit_rows[name].fillna(0).sum()) for name in names}}
        del model
    _progress("bootstrap", queries=audit_queries)
    paired = _cluster_intervals(audit_rows, scores["manual"], scores["automatic"])
    midpoint = audit_rows[["query_id", "event_time"]].drop_duplicates()["event_time"].quantile(0.5)
    halves = {}
    for label, mask in (("earlier_half", audit_rows["event_time"] <= midpoint),
                        ("later_half", audit_rows["event_time"] > midpoint)):
        halves[label] = _cluster_intervals(audit_rows.loc[mask], scores["manual"][mask], scores["automatic"][mask])
    np.savez_compressed(output_dir / "audit-scores.npz", manual=scores["manual"], automatic=scores["automatic"])
    connection = _connection()
    try:
        connection.register("audit_labels", audit_rows[["user_id", "query_id", "item_id", "event_time", "label"]])
        connection.execute(f"COPY audit_labels TO {quote_sql_string(str(output_dir / 'audit-labels.parquet'))} (FORMAT PARQUET, COMPRESSION ZSTD)")
    finally:
        connection.close()
    return {"cutoffs": [str(value) for value in cutoffs], "audit_queries": audit_queries,
            "audit_users": audit_users, "audit_positives": int(audit_positives),
            "rows": {"train": int(train.sum()), "selection": int(selection.sum()), "audit": int(audit.sum())},
            "union_feature_generation_seconds": feature_seconds,
            "models": results, "paired": paired, "audit_halves": halves}


def _recover_saved_quality(manifest: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Resume after an export failure without fitting or selecting models again."""
    connection = _connection()
    try:
        candidate_sql = quote_sql_string(manifest["candidates"])
        audit_sql = quote_sql_string(manifest["audit_candidates"])
        cutoffs = connection.execute(f"""
            SELECT quantile_disc(timestamp, .5), quantile_disc(timestamp, .85), MAX(timestamp)
            FROM (SELECT DISTINCT query_id, timestamp FROM read_parquet({candidate_sql}))
        """).fetchone()
        queries = connection.execute(f"SELECT DISTINCT query_id, timestamp FROM read_parquet({audit_sql}) ORDER BY timestamp, query_id").fetchdf()[["query_id"]]
        partition_rows = connection.execute(f"""
            SELECT COUNT(*) FILTER (WHERE timestamp <= ?),
                   COUNT(*) FILTER (WHERE timestamp > ? AND timestamp <= ?),
                   COUNT(*) FILTER (WHERE timestamp > ?)
            FROM read_parquet({candidate_sql})
        """, [cutoffs[0], cutoffs[0], cutoffs[1], cutoffs[1]]).fetchone()
    finally:
        connection.close()
    specs = {spec.name: spec for variant in ("manual", "automatic") for spec in _load_specs(manifest, variant)}
    frames = []
    for offset in range(0, len(queries), 2000):
        connection = _connection()
        try:
            _examples(connection, Path(manifest["audit_candidates"]), queries.iloc[offset:offset + 2000])
            build_generated_feature_table(connection, Path(manifest["event_history"]), tuple(specs.values()))
            frames.append(connection.execute("SELECT * FROM generated_model_data ORDER BY row_id").fetchdf())
        finally:
            connection.close()
    rows = pd.concat(frames, ignore_index=True)
    for column in CATEGORICAL_FEATURES:
        rows[column] = rows[column].fillna("__MISSING__").astype(str)
    original_scores = np.load(output_dir / "audit-scores.npz")
    results, scores = {}, {}
    for variant in ("manual", "automatic"):
        names = tuple(spec.name for spec in _load_specs(manifest, variant))
        model_path = output_dir / f"{variant}.cbm"
        model = catboost.CatBoostClassifier()
        model.load_model(str(model_path))
        if model.feature_names_ != [*CATEGORICAL_FEATURES, *names]:
            raise DataContractError("Saved model features differ from the frozen manifest.")
        scores[variant] = model.predict_proba(rows[[*CATEGORICAL_FEATURES, *names]])[:, 1]
        if not np.allclose(np.sort(scores[variant]), np.sort(original_scores[variant]), rtol=1e-10, atol=1e-12):
            raise DataContractError("Recovered prediction multiset differs from the completed original fit.")
        result = {"iterations": model.tree_count_, "fit_seconds": None, "rows": len(rows),
                  "metrics": evaluate_predictions(rows, pd.Series(scores[variant], index=rows.index))}
        results[variant] = {"selection": {"iterations": model.tree_count_, "metrics": None},
                            "audit": result, "model_bytes": model_path.stat().st_size,
                            "audit_feature_sums": {name: float(rows[name].fillna(0).sum()) for name in names},
                            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()}
    midpoint = rows[["query_id", "event_time"]].drop_duplicates()["event_time"].quantile(.5)
    halves = {}
    for label, mask in (("earlier_half", rows["event_time"] <= midpoint),
                        ("later_half", rows["event_time"] > midpoint)):
        halves[label] = _cluster_intervals(rows.loc[mask], scores["manual"][mask], scores["automatic"][mask])
    connection = _connection()
    try:
        connection.register("audit_labels", rows[["user_id", "query_id", "item_id", "event_time", "label"]])
        connection.execute(f"COPY audit_labels TO {quote_sql_string(str(output_dir / 'audit-labels.parquet'))} (FORMAT PARQUET, COMPRESSION ZSTD)")
    finally:
        connection.close()
    np.savez_compressed(output_dir / "recovered-audit-scores.npz", **scores)
    return {"cutoffs": [str(value) for value in cutoffs], "audit_queries": len(queries),
            "audit_users": int(rows["user_id"].nunique()), "audit_positives": int(rows["label"].sum()),
            "rows": dict(zip(("train", "selection", "audit"), partition_rows)),
            "models": results, "paired": _cluster_intervals(rows, scores["manual"], scores["automatic"]),
            "audit_halves": halves,
            "recovery": "Previously completed frozen models; predictions match original multisets. Fit durations and selection metrics were lost on Parquet export failure and are not reconstructed."}


def _benchmarks(manifest_path: Path, output_dir: Path, quality: dict[str, Any]) -> dict[str, Any]:
    measured: dict[str, list[dict[str, Any]]] = {"manual": [], "automatic": []}
    for repeat in range(6):
        order = ("manual", "automatic") if repeat % 2 == 0 else ("automatic", "manual")
        for variant in order:
            _progress("benchmark", repeat=repeat, variant=variant, warmup=repeat == 0)
            result_path = output_dir / "measurements" / f"{repeat}-{variant}.json"
            subprocess.run([
                sys.executable, "-m", "recsys_mvp.feature_efficiency", "--worker",
                "--manifest", str(manifest_path), "--variant", variant,
                "--worker-output", str(result_path),
            ], check=True)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result["rows"] != quality["rows"]["audit"]:
                raise DataContractError("Benchmark and quality matrices have different row counts.")
            for name, expected in quality["models"][variant]["audit_feature_sums"].items():
                if not np.isclose(result["feature_sums"][name], expected, rtol=1e-9, atol=1e-6):
                    raise DataContractError(f"Benchmark feature differs from quality matrix: {name}")
            if repeat:
                measured[variant].append(result)
    summaries = {}
    for variant, runs in measured.items():
        summaries[variant] = {
            "measured_runs": len(runs),
            "median_build_and_fetch_seconds": statistics.median(row["build_and_fetch_seconds"] for row in runs),
            "median_materialize_seconds": statistics.median(row["materialize_seconds"] for row in runs),
            "median_peak_rss_mb": statistics.median(row["peak_rss_mb"] for row in runs),
            "min_build_and_fetch_seconds": min(row["build_and_fetch_seconds"] for row in runs),
            "max_build_and_fetch_seconds": max(row["build_and_fetch_seconds"] for row in runs),
            "numeric_matrix_bytes": runs[0]["numeric_matrix_bytes"],
            "runs": runs,
        }
    manual, automatic = summaries["manual"], summaries["automatic"]
    savings = manual["median_build_and_fetch_seconds"] - automatic["median_build_and_fetch_seconds"]
    return {"variants": summaries,
            "time_saving_fraction": savings / manual["median_build_and_fetch_seconds"],
            "peak_rss_change_fraction": automatic["median_peak_rss_mb"] / manual["median_peak_rss_mb"] - 1,
            "conditional_recalculations_to_amortize_previous_search": 148.755 / savings if savings > 0 else None}


def _finish(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir = manifest_path.parent
    preparation = json.loads((output_dir / "preparation.json").read_text(encoding="utf-8"))
    quality_path = output_dir / "quality.json"
    if quality_path.exists():
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
    elif all((output_dir / name).exists() for name in ("manual.cbm", "automatic.cbm", "audit-scores.npz")):
        _progress("recover_saved_models")
        quality = _recover_saved_quality(manifest, output_dir)
        _write_json(quality_path, quality)
    else:
        quality = _quality(manifest, output_dir)
        _write_json(quality_path, quality)
    benchmark = _benchmarks(manifest_path, output_dir, quality)
    checks = {"time": benchmark["time_saving_fraction"] >= 0.20,
              "memory": benchmark["peak_rss_change_fraction"] <= 0.05,
              "quality_noninferiority": quality["paired"]["noninferiority_margin"]["bootstrap_95_percent_interval"][0] > 0}
    report = {"manifest": manifest, "preparation": preparation, "quality": quality,
              "benchmark": benchmark, "criteria_passed": checks,
              "joint_success": all(checks.values()), "previous_query_overlap": 0}
    _write_json(output_dir / "summary.json", report)
    _progress("complete", criteria_passed=checks, joint_success=report["joint_success"])
    return report


def run_feature_efficiency(project_dir: Path, output_dir: Path, resume: bool = False) -> dict[str, Any]:
    project_dir, output_dir = project_dir.resolve(), output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "manifest.json").exists():
        if resume and not (output_dir / "summary.json").exists():
            manifest_path = output_dir / "manifest.json"
            frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
            protocol = project_dir / "experiments/feature-efficiency-protocol.md"
            if hashlib.sha256(protocol.read_bytes()).hexdigest() != frozen["protocol_sha256"]:
                raise DataContractError("The frozen protocol changed; resume is not allowed.")
            _progress("resume", candidates=frozen["candidates"])
            return _finish(manifest_path)
        raise DataContractError("Output directory already has a frozen experiment; choose a new directory.")
    selected_path = project_dir / "experiments/feature-efficiency-selection.json"
    previous = json.loads(selected_path.read_text(encoding="utf-8"))
    selected = tuple(previous["selected_features"])
    formula_map = {
        spec.name: spec.as_dict()
        for spec in generate_feature_specs(("user", "item"))
    }
    if len(selected) != 8 or len(set(selected)) != 8:
        raise DataContractError("Expected the previously fixed eight-feature selection.")
    excluded_inputs = tuple(
        project_dir / "data/processed" / name
        for name in previous["excluded_candidate_files"]
    )
    missing_inputs = [str(path) for path in excluded_inputs if not path.is_file()]
    if missing_inputs:
        raise DataContractError(
            "Reproduce the prior candidate files before running the frozen comparison: "
            + ", ".join(missing_inputs)
        )
    protocol = project_dir / "experiments/feature-efficiency-protocol.md"
    candidates = output_dir / "candidates.parquet"
    manual_specs = {spec.name: spec for spec in _manual_feature_specs()}
    manifest = {
        "frozen_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "selected_source": str(selected_path),
        "selected_source_sha256": hashlib.sha256(selected_path.read_bytes()).hexdigest(),
        "feature_specs": {"manual": [manual_specs[name].as_dict() for name in MANUAL_FEATURE_NAMES],
                          "automatic": [formula_map[name] for name in selected]},
        "candidates": str(candidates), "audit_candidates": str(output_dir / "audit-candidates.parquet"),
        "event_history": str(project_dir / "data/processed/yambda_events.parquet"),
        "excluded_query_inputs": [str(path) for path in excluded_inputs],
        "criteria": {"minimum_time_saving": 0.20, "maximum_rss_increase": 0.05, "relative_ndcg_loss_margin": 0.02},
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "catboost": catboost.__version__, "duckdb": duckdb.__version__,
                        "pandas": pd.__version__, "numpy": np.__version__},
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    _progress("prepare_candidates", query_seed=48, query_hash_modulo=100)
    preparation = prepare_yambda_candidates(
        project_dir / "data/raw/yambda-50m-multi_event.parquet", candidates,
        output_dir / "positive-history.parquet", query_sample_modulo=100,
        negatives_per_query=50, seed=42, negative_strategy="mixed", popular_pool_size=50000,
        query_seed=48, popular_pool_cutoff_seconds=6524410, catalog_cutoff_seconds=21381550,
        unseen_positive_only=True, excluded_query_inputs=excluded_inputs,
    )
    _write_json(output_dir / "preparation.json", preparation)
    connection = _connection()
    try:
        paths_sql = ", ".join(quote_sql_string(str(path)) for path in excluded_inputs)
        overlap = connection.execute(f"""
            SELECT COUNT(DISTINCT fresh.query_id) FROM read_parquet({quote_sql_string(str(candidates))}) AS fresh
            JOIN (SELECT DISTINCT query_id FROM read_parquet([{paths_sql}])) AS old USING (query_id)
        """).fetchone()[0]
        if overlap:
            raise DataContractError("Fresh candidate queries overlap prior experiment queries.")
    finally:
        connection.close()
    _progress("candidates_ready", overlap_with_previous=overlap)
    return _finish(manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed automatic features for quality and recurring batch cost.")
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/feature-efficiency"))
    parser.add_argument("--resume", action="store_true", help="Resume an unfinished run with its frozen manifest and candidates.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--variant", choices=("manual", "automatic"))
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.manifest is None or args.variant is None or args.worker_output is None:
            parser.error("Worker requires manifest, variant and worker-output.")
        _benchmark_worker(args.manifest, args.variant, args.worker_output)
    else:
        run_feature_efficiency(args.project_dir, args.output_dir, args.resume)


if __name__ == "__main__":
    main()

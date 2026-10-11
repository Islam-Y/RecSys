from __future__ import annotations

import json
import math
import resource
import sys
import time
from pathlib import Path
from typing import Any

import catboost
import duckdb
import numpy as np
import pandas as pd

from recsys_mvp.auto_features import (
    SEARCH_WINDOWS_DAYS,
    FeatureSpec,
    build_generated_feature_table,
    generate_feature_specs,
)
from recsys_mvp.data import DataContractError, create_examples_view, register_source
from recsys_mvp.evaluation import evaluate_predictions, evaluate_ranking_scores


MANUAL_FEATURE_NAMES = (
    *(f"user_listen_plus_count_{days}d" for days in (1, 7, 30, 90)),
    *(f"item_listen_plus_count_{days}d" for days in (1, 7, 30, 90)),
    *(f"user_listen_plus_distinct_items_{days}d" for days in (30, 90)),
    *(f"item_listen_plus_distinct_users_{days}d" for days in (30, 90)),
)
CATEGORICAL_FEATURES = ("user_id", "item_id")


def _prepare_examples(
    connection: duckdb.DuckDBPyConnection,
    candidate_path: Path,
    split_quantiles: tuple[float, float, float],
    audit_cutoff: Any | None,
) -> tuple[Any, Any, Any]:
    columns = register_source(connection, candidate_path)
    create_examples_view(
        connection, columns, "user_id", "item_id", "timestamp", "label", "query_id"
    )
    cutoffs = connection.execute(
        f"""
        SELECT quantile_disc(event_time, {split_quantiles[0]}),
               quantile_disc(event_time, {split_quantiles[1]}),
               quantile_disc(event_time, {split_quantiles[2]})
        FROM examples
        """
    ).fetchone()
    if any(cutoff is None for cutoff in cutoffs):
        raise DataContractError("The candidate table has no valid temporal split.")
    if audit_cutoff is not None and cutoffs[2] != audit_cutoff:
        raise DataContractError("Candidate split changed between feature builds.")
    connection.execute(
        "CREATE TEMP TABLE selected_examples AS SELECT * FROM examples WHERE event_time <= ?",
        [cutoffs[2]],
    )
    connection.execute("DROP VIEW examples")
    connection.execute("CREATE TEMP VIEW examples AS SELECT * FROM selected_examples")
    return cutoffs


def _build_matrix(
    candidate_path: Path,
    event_history_path: Path,
    specs: tuple[FeatureSpec, ...],
    split_quantiles: tuple[float, float, float],
    audit_cutoff: Any | None = None,
) -> tuple[pd.DataFrame, tuple[Any, Any, Any], float]:
    connection = duckdb.connect()
    try:
        cutoffs = _prepare_examples(
            connection, candidate_path, split_quantiles, audit_cutoff
        )
        started = time.perf_counter()
        build_generated_feature_table(connection, event_history_path, specs)
        feature_seconds = time.perf_counter() - started
        rows = connection.execute(
            "SELECT * FROM generated_model_data ORDER BY row_id"
        ).fetchdf()
        return rows, cutoffs, round(feature_seconds, 3)
    finally:
        connection.close()


def _make_model(iterations: int) -> catboost.CatBoostClassifier:
    return catboost.CatBoostClassifier(
        iterations=iterations,
        depth=8,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=4,
    )


def _fit_and_score(
    rows: pd.DataFrame,
    feature_names: tuple[str, ...],
    train_mask: pd.Series,
    score_mask: pd.Series,
    fixed_iterations: int | None = None,
) -> tuple[catboost.CatBoostClassifier, dict[str, Any]]:
    columns = [*CATEGORICAL_FEATURES, *feature_names]
    train_rows = rows.loc[train_mask]
    score_rows = rows.loc[score_mask]
    model = _make_model(fixed_iterations or 500)
    fit_kwargs: dict[str, Any] = {}
    if fixed_iterations is None:
        fit_kwargs = {
            "eval_set": (
                score_rows[columns],
                score_rows["label"].astype(int),
            ),
            "early_stopping_rounds": 50,
            "use_best_model": True,
        }
    started = time.perf_counter()
    model.fit(
        train_rows[columns],
        train_rows["label"].astype(int),
        cat_features=list(CATEGORICAL_FEATURES),
        **fit_kwargs,
    )
    fit_seconds = round(time.perf_counter() - started, 3)
    scores = model.predict_proba(score_rows[columns])[:, 1]
    metrics = evaluate_predictions(
        score_rows,
        pd.Series(scores, index=score_rows.index),
    )
    return model, {
        "features": list(feature_names),
        "numeric_feature_count": len(feature_names),
        "iterations": fixed_iterations or model.get_best_iteration() + 1,
        "fit_seconds": fit_seconds,
        "rows": len(score_rows),
        "metrics": metrics,
    }


def _paired_ndcg_delta(
    frame: pd.DataFrame,
    manual_scores: np.ndarray,
    selected_scores: np.ndarray,
    k: int = 3,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    labels = frame["label"].astype(int).to_numpy()
    deltas = []
    for positions in frame.groupby(
        ["user_id", "query_id"], sort=False
    ).indices.values():
        query_labels = labels[positions]
        positives = int(query_labels.sum())
        if len(positions) < 2 or positives == 0:
            continue
        idcg = sum(1.0 / math.log2(rank + 2) for rank in range(min(positives, k)))

        def ndcg(scores: np.ndarray) -> float:
            order = np.argsort(-scores[positions], kind="stable")[:k]
            dcg = sum(
                float(query_labels[index]) / math.log2(rank + 2)
                for rank, index in enumerate(order)
            )
            return dcg / idcg

        deltas.append(ndcg(selected_scores) - ndcg(manual_scores))
    if not deltas:
        raise DataContractError("No evaluable queries for paired NDCG comparison.")
    values = np.asarray(deltas, dtype=float)
    generator = np.random.default_rng(42)
    samples = generator.choice(values, size=(bootstrap_samples, len(values)), replace=True)
    lower, upper = np.quantile(samples.mean(axis=1), [0.025, 0.975])
    return {
        "selected_minus_manual_ndcg@3": float(values.mean()),
        "bootstrap_95_percent_interval": [float(lower), float(upper)],
        "queries": len(values),
        "resamples": bootstrap_samples,
        "seed": 42,
    }


def run_feature_search(
    candidate_path: Path,
    event_history_path: Path,
    output_dir: Path,
    entities: tuple[str, ...] = ("user", "item"),
    windows_days: tuple[int, ...] = SEARCH_WINDOWS_DAYS,
    feature_budget: int = 12,
    split_quantiles: tuple[float, float, float] = (0.50, 0.70, 0.85),
) -> dict[str, Any]:
    if feature_budget < 1:
        raise DataContractError("Feature budget must be positive.")
    if not 0 < split_quantiles[0] < split_quantiles[1] < split_quantiles[2] < 1:
        raise DataContractError("Split quantiles must increase and stay within 0-1.")
    candidate_path = candidate_path.expanduser().resolve()
    event_history_path = event_history_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    specs = generate_feature_specs(entities, windows_days)
    spec_by_name = {spec.name: spec for spec in specs}
    missing_manual = set(MANUAL_FEATURE_NAMES) - set(spec_by_name)
    if missing_manual:
        raise DataContractError(
            "The search space must include the manual control features: "
            f"{sorted(missing_manual)}"
        )
    if feature_budget < len(MANUAL_FEATURE_NAMES):
        raise DataContractError(
            f"Feature budget must be at least {len(MANUAL_FEATURE_NAMES)} "
            "to compare with the fixed manual control."
        )

    started = time.perf_counter()
    rows, cutoffs, all_feature_seconds = _build_matrix(
        candidate_path, event_history_path, specs, split_quantiles
    )
    for column in CATEGORICAL_FEATURES:
        rows[column] = rows[column].fillna("__MISSING__").astype(str)
    train_mask = rows["event_time"] <= cutoffs[0]
    selection_mask = (rows["event_time"] > cutoffs[0]) & (
        rows["event_time"] <= cutoffs[1]
    )
    audit_mask = rows["event_time"] > cutoffs[1]
    if not train_mask.any() or not selection_mask.any() or not audit_mask.any():
        raise DataContractError("Temporal feature-search split has an empty partition.")
    if rows.loc[train_mask, "label"].nunique() < 2:
        raise DataContractError("Training partition must contain both labels.")

    numeric_pool = tuple(
        spec.name
        for spec in specs
        if rows.loc[train_mask, spec.name].nunique(dropna=False) > 1
    )
    if len(numeric_pool) < feature_budget:
        raise DataContractError(
            "Too few nonconstant generated features for the requested budget."
        )
    if set(MANUAL_FEATURE_NAMES) - set(numeric_pool):
        raise DataContractError("A manual control feature is constant on training.")

    full_model, full_selection = _fit_and_score(
        rows, numeric_pool, train_mask, selection_mask
    )
    importance = dict(
        zip(
            [*CATEGORICAL_FEATURES, *numeric_pool],
            map(float, full_model.get_feature_importance()),
        )
    )
    ranked = tuple(
        sorted(numeric_pool, key=lambda name: (-importance[name], name))
    )
    _, manual_selection = _fit_and_score(
        rows, MANUAL_FEATURE_NAMES, train_mask, selection_mask
    )
    selection_runs: list[dict[str, Any]] = []
    selected_run = None
    candidate_budgets = sorted(
        {4, 8, feature_budget} & set(range(1, feature_budget + 1))
    )
    for budget in candidate_budgets:
        _, result = _fit_and_score(
            rows, ranked[:budget], train_mask, selection_mask
        )
        selection_runs.append(result)
        key = (
            result["metrics"]["ndcg@3"],
            result["metrics"]["recall@3"],
            -result["numeric_feature_count"],
        )
        if selected_run is None or key > (
            selected_run["metrics"]["ndcg@3"],
            selected_run["metrics"]["recall@3"],
            -selected_run["numeric_feature_count"],
        ):
            selected_run = result
    if selected_run is None:
        raise DataContractError("Feature search found no candidate feature set.")

    train_and_selection_mask = rows["event_time"] <= cutoffs[1]
    manual_audit_model, manual_audit = _fit_and_score(
        rows,
        MANUAL_FEATURE_NAMES,
        train_and_selection_mask,
        audit_mask,
        manual_selection["iterations"],
    )
    selected_names = tuple(selected_run["features"])
    selected_audit_model, selected_audit = _fit_and_score(
        rows,
        selected_names,
        train_and_selection_mask,
        audit_mask,
        selected_run["iterations"],
    )
    audit_rows = rows.loc[audit_mask]
    manual_scores = manual_audit_model.predict_proba(
        audit_rows[[*CATEGORICAL_FEATURES, *MANUAL_FEATURE_NAMES]]
    )[:, 1]
    selected_scores = selected_audit_model.predict_proba(
        audit_rows[[*CATEGORICAL_FEATURES, *selected_names]]
    )[:, 1]
    paired_delta = _paired_ndcg_delta(
        audit_rows, manual_scores, selected_scores
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manual_audit_model.save_model(str(output_dir / "manual.cbm"))
    selected_audit_model.save_model(str(output_dir / "selected.cbm"))

    manual_specs = tuple(spec_by_name[name] for name in MANUAL_FEATURE_NAMES)
    _, manual_cutoffs, manual_feature_seconds = _build_matrix(
        candidate_path, event_history_path, manual_specs, split_quantiles, cutoffs[2]
    )
    if manual_cutoffs != cutoffs:
        raise DataContractError("Manual and generated runs used different cutoffs.")
    selected_specs = tuple(spec_by_name[name] for name in selected_names)
    _, selected_cutoffs, selected_feature_seconds = _build_matrix(
        candidate_path, event_history_path, selected_specs, split_quantiles, cutoffs[2]
    )
    if selected_cutoffs != cutoffs:
        raise DataContractError("Selected and generated runs used different cutoffs.")
    popularity_audit = evaluate_ranking_scores(
        rows.loc[audit_mask],
        rows.loc[audit_mask, "label"].astype(int),
        rows.loc[audit_mask, "item_listen_plus_count_30d"].astype(float),
        k=3,
    )
    peak_value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    report = {
        "protocol": {
            "candidate_input": str(candidate_path),
            "event_history_input": str(event_history_path),
            "split_quantiles": list(split_quantiles),
            "cutoffs": [str(cutoff) for cutoff in cutoffs],
            "rows": {
                "train": int(train_mask.sum()),
                "selection": int(selection_mask.sum()),
                "audit": int(audit_mask.sum()),
            },
            "history_rule": "event_time strictly before candidate timestamp",
            "test_metrics_evaluated": False,
            "selection_rule": (
                "rank nonconstant numeric formulas by full-model feature importance "
                "on train/selection; compare ranked prefixes on selection NDCG@3, "
                "Recall@3 tie-breaker; refit manual and chosen prefixes on "
                "train+selection with iteration counts chosen on selection"
            ),
            "feature_budget": feature_budget,
            "entities": list(entities),
            "event_kinds": ["listen_plus", "listen_short", "like", "dislike"],
            "windows_days": list(windows_days),
            "model": "CatBoostClassifier depth=8, learning_rate=0.05, seed=42",
            "candidate_semantics": "sampled unknowns, not observed non-clicks",
        },
        "generated_formulas": [spec.as_dict() for spec in specs],
        "nonconstant_formula_count": len(numeric_pool),
        "feature_importance": importance,
        "full_pool_selection": full_selection,
        "manual": {
            "selection": manual_selection,
            "audit": manual_audit,
            "feature_generation_seconds": manual_feature_seconds,
        },
        "automatic": {
            "selection_runs": selection_runs,
            "chosen_selection": selected_run,
            "audit": selected_audit,
            "all_feature_generation_seconds": all_feature_seconds,
            "selected_feature_generation_seconds": selected_feature_seconds,
        },
        "item_popularity_30d_audit": popularity_audit,
        "paired_audit_comparison": paired_delta,
        "runtime": {
            "total_seconds": round(time.perf_counter() - started, 3),
            "peak_process_memory_mb": round(peak_value / divisor, 1),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report

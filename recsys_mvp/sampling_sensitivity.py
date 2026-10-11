from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from recsys_mvp.auto_features import SEARCH_WINDOWS_DAYS, generate_feature_specs
from recsys_mvp.data import DataContractError, quote_sql_string
from recsys_mvp.evaluation import evaluate_predictions
from recsys_mvp.feature_search import (
    CATEGORICAL_FEATURES,
    MANUAL_FEATURE_NAMES,
    _build_matrix,
    _fit_and_score,
)
from recsys_mvp.robust_search import _masks


def _sampling_metadata(candidate_path: Path) -> dict[str, Any]:
    metadata_path = candidate_path.with_suffix(
        candidate_path.suffix + ".metadata.json"
    )
    if not metadata_path.is_file():
        raise DataContractError(f"Candidate metadata is missing: {metadata_path}")
    candidate_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_path = Path(candidate_metadata.get("source", ""))
    source_metadata_path = source_path.with_suffix(
        source_path.suffix + ".metadata.json"
    )
    if not source_metadata_path.is_file():
        raise DataContractError(
            f"Original candidate metadata is missing: {source_metadata_path}"
        )
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    sampling = source_metadata.get("task", {}).get("sampling", {})
    if "negative_sampling_seed" not in sampling and "seed" in sampling:
        sampling["negative_sampling_seed"] = sampling["seed"]
    required = {
        "negative_sampling_seed",
        "query_seed",
        "query_hash_modulo",
        "requested_negatives_per_query",
        "negative_strategy",
        "popular_pool_size_requested",
        "popular_pool_cutoff_event_time_seconds",
        "catalog_cutoff_event_time_seconds",
    }
    if missing := required - set(sampling):
        raise DataContractError(
            f"Sampling metadata for {candidate_path} is missing: {sorted(missing)}"
        )
    return sampling


def _query_ids(path: Path) -> set[str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise DataContractError(f"Candidate file does not exist: {resolved}")
    connection = duckdb.connect()
    try:
        rows = connection.execute(
            f"SELECT DISTINCT query_id FROM read_parquet({quote_sql_string(str(resolved))})"
        ).fetchall()
        return {str(row[0]) for row in rows}
    finally:
        connection.close()


def _positive_signature(path: Path) -> tuple[tuple[Any, ...], ...]:
    resolved = path.expanduser().resolve()
    connection = duckdb.connect()
    try:
        rows = connection.execute(
            f"""
            SELECT DISTINCT query_id, user_id, item_id, timestamp
            FROM read_parquet({quote_sql_string(str(resolved))})
            WHERE label = 1
            ORDER BY query_id, user_id, item_id, timestamp
            """
        ).fetchall()
        return tuple(tuple(row) for row in rows)
    finally:
        connection.close()


def _negative_pairs(path: Path) -> set[tuple[str, str]]:
    resolved = path.expanduser().resolve()
    connection = duckdb.connect()
    try:
        rows = connection.execute(
            f"""
            SELECT DISTINCT query_id, item_id
            FROM read_parquet({quote_sql_string(str(resolved))})
            WHERE label = 0
            """
        ).fetchall()
        return {(str(query_id), str(item_id)) for query_id, item_id in rows}
    finally:
        connection.close()


def _filter_queries(
    source: Path,
    destination: Path,
    excluded_query_ids: set[str],
) -> dict[str, int]:
    source = source.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    connection = duckdb.connect()
    try:
        excluded = pd.DataFrame({"query_id": sorted(excluded_query_ids)})
        connection.register("excluded_queries", excluded)
        before = connection.execute(
            f"SELECT COUNT(DISTINCT query_id) FROM read_parquet({quote_sql_string(str(source))})"
        ).fetchone()[0]
        connection.execute(
            f"""
            COPY (
                SELECT candidate.*
                FROM read_parquet({quote_sql_string(str(source))}) AS candidate
                ANTI JOIN excluded_queries USING (query_id)
            ) TO {quote_sql_string(str(destination))}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        after = connection.execute(
            f"SELECT COUNT(DISTINCT query_id) FROM read_parquet({quote_sql_string(str(destination))})"
        ).fetchone()[0]
        rows, positives = connection.execute(
            f"SELECT COUNT(*), SUM(label) FROM read_parquet({quote_sql_string(str(destination))})"
        ).fetchone()
        return {
            "queries_before": int(before),
            "queries_removed": int(before - after),
            "queries_after": int(after),
            "rows_after": int(rows),
            "positives_after": int(positives or 0),
        }
    finally:
        connection.close()


def _query_ndcg_values(frame: pd.DataFrame, scores: np.ndarray, k: int = 3):
    labels = frame["label"].astype(int).to_numpy()
    groups = frame.groupby(["user_id", "query_id"], sort=False).indices
    values: dict[tuple[str, str], float] = {}
    for key, positions in groups.items():
        query_labels = labels[positions]
        positives = int(query_labels.sum())
        if len(positions) < 2 or positives == 0:
            continue
        ideal_dcg = sum(
            1.0 / math.log2(rank + 2) for rank in range(min(positives, k))
        )
        order = np.argsort(-scores[positions], kind="stable")[:k]
        dcg = sum(
            float(query_labels[index]) / math.log2(rank + 2)
            for rank, index in enumerate(order)
        )
        values[(str(key[0]), str(key[1]))] = dcg / ideal_dcg
    return values


def _query_recall_values(frame: pd.DataFrame, scores: np.ndarray, k: int = 3):
    labels = frame["label"].astype(int).to_numpy()
    groups = frame.groupby(["user_id", "query_id"], sort=False).indices
    values: dict[tuple[str, str], float] = {}
    for key, positions in groups.items():
        query_labels = labels[positions]
        positives = int(query_labels.sum())
        if len(positions) < 2 or positives == 0:
            continue
        order = np.argsort(-scores[positions], kind="stable")[:k]
        values[(str(key[0]), str(key[1]))] = float(
            query_labels[order].sum() / positives
        )
    return values


def _paired_bootstrap(
    reference: dict[tuple[str, str], float],
    comparison: dict[tuple[str, str], float],
    bootstrap_samples: int = 2000,
    metric_name: str = "comparison_minus_reference_ndcg@3",
) -> dict[str, Any]:
    common = sorted(reference.keys() & comparison.keys())
    if not common:
        raise DataContractError("No shared queries for paired sample comparison.")
    deltas_by_user: dict[str, list[float]] = {}
    for user_id, query_id in common:
        deltas_by_user.setdefault(user_id, []).append(
            comparison[(user_id, query_id)] - reference[(user_id, query_id)]
        )
    deltas = np.asarray(
        [delta for user_deltas in deltas_by_user.values() for delta in user_deltas],
        dtype=float,
    )
    rng = np.random.default_rng(42)
    user_ids = tuple(deltas_by_user)
    bootstrap_means = np.empty(bootstrap_samples, dtype=float)
    for index in range(bootstrap_samples):
        sampled_users = rng.choice(user_ids, size=len(user_ids), replace=True)
        sampled_deltas = [
            delta
            for user_id in sampled_users
            for delta in deltas_by_user[str(user_id)]
        ]
        bootstrap_means[index] = float(np.mean(sampled_deltas))
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        metric_name: float(deltas.mean()),
        "bootstrap_95_percent_interval": [float(lower), float(upper)],
        "paired_queries": len(common),
        "paired_users": len(deltas_by_user),
        "resamples": bootstrap_samples,
        "seed": 42,
        "bootstrap_unit": "user; all queries for a sampled user stay together",
    }


def _check_shared_sampling_config(
    variants: tuple[tuple[str, Path], ...], vary_key: str
) -> dict[str, Any]:
    if len(variants) < 2:
        raise DataContractError("Each sensitivity group needs at least two variants.")
    configs = [(label, path, _sampling_metadata(path)) for label, path in variants]
    invariant_keys = (
        "query_hash_modulo",
        "requested_negatives_per_query",
        "negative_strategy",
        "popular_pool_size_requested",
        "popular_pool_cutoff_event_time_seconds",
        "catalog_cutoff_event_time_seconds",
    )
    baseline = configs[0][2]
    for label, _, config in configs[1:]:
        if any(config[key] != baseline[key] for key in invariant_keys):
            raise DataContractError(
                f"Sampling configuration differs for variant {label}."
            )
    varied_values = [config[vary_key] for _, _, config in configs]
    if len(set(varied_values)) != len(varied_values):
        raise DataContractError(f"Each variant must have a distinct {vary_key}.")
    fixed_key = (
        "query_seed" if vary_key == "negative_sampling_seed" else "negative_sampling_seed"
    )
    if len({config[fixed_key] for _, _, config in configs}) != 1:
        raise DataContractError(f"Sensitivity group must hold {fixed_key} fixed.")
    return {
        "varies": vary_key,
        "fixed_value": configs[0][2][fixed_key],
        "variants": [
            {"label": label, "input": str(path), "sampling": config}
            for label, path, config in configs
        ],
    }


def _fit_sample(
    label: str,
    candidate_path: Path,
    event_history_path: Path,
    specs,
    selected_features: tuple[str, ...],
):
    rows, cutoffs, feature_seconds = _build_matrix(
        candidate_path,
        event_history_path,
        specs,
        (0.50, 0.70, 0.85),
    )
    for column in CATEGORICAL_FEATURES:
        rows[column] = rows[column].fillna("__MISSING__").astype(str)
    rows = rows.sort_values(
        ["event_time", "query_id", "label", "item_id"],
        ascending=[True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    train_mask, selection_mask, audit_mask = _masks(rows, cutoffs)
    train_and_selection = train_mask | selection_mask
    audit_rows = rows.loc[audit_mask]
    audit_queries = set(audit_rows["query_id"].astype(str).unique())
    if not audit_queries:
        raise DataContractError(f"No audit queries in sample {label}.")

    _, manual_selection = _fit_and_score(
        rows, MANUAL_FEATURE_NAMES, train_mask, selection_mask
    )
    _, automatic_selection = _fit_and_score(
        rows, selected_features, train_mask, selection_mask
    )
    manual_model, manual_audit = _fit_and_score(
        rows,
        MANUAL_FEATURE_NAMES,
        train_and_selection,
        audit_mask,
        manual_selection["iterations"],
    )
    automatic_model, automatic_audit = _fit_and_score(
        rows,
        selected_features,
        train_and_selection,
        audit_mask,
        automatic_selection["iterations"],
    )
    manual_scores = manual_model.predict_proba(
        audit_rows[[*CATEGORICAL_FEATURES, *MANUAL_FEATURE_NAMES]]
    )[:, 1]
    automatic_scores = automatic_model.predict_proba(
        audit_rows[[*CATEGORICAL_FEATURES, *selected_features]]
    )[:, 1]
    manual_query_ndcg = _query_ndcg_values(audit_rows, manual_scores)
    automatic_query_ndcg = _query_ndcg_values(audit_rows, automatic_scores)
    paired = _paired_bootstrap(
        manual_query_ndcg,
        automatic_query_ndcg,
        metric_name="selected_minus_manual_ndcg@3",
    )
    return {
        "label": label,
        "input": str(candidate_path),
        "cutoffs": [str(value) for value in cutoffs],
        "rows": {
            "train": int(train_mask.sum()),
            "selection": int(selection_mask.sum()),
            "audit": int(audit_mask.sum()),
        },
        "audit_query_count": len(audit_queries),
        "feature_generation_seconds": feature_seconds,
        "manual": {
            "selection": manual_selection,
            "audit": manual_audit,
        },
        "automatic": {
            "selection": automatic_selection,
            "audit": automatic_audit,
        },
        "paired_automatic_minus_manual": paired,
        "audit_query_ndcg": {
            "manual": manual_query_ndcg,
            "automatic": automatic_query_ndcg,
        },
    }


def _serialize_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in sample.items() if key != "audit_query_ndcg"}


def _evaluate_model(
    model,
    rows: pd.DataFrame,
    feature_names: tuple[str, ...],
) -> tuple[
    dict[str, Any],
    dict[tuple[str, str], float],
    dict[tuple[str, str], float],
]:
    scores = model.predict_proba(
        rows[[*CATEGORICAL_FEATURES, *feature_names]]
    )[:, 1]
    metrics = evaluate_predictions(
        rows, pd.Series(scores, index=rows.index)
    )
    return (
        metrics,
        _query_ndcg_values(rows, scores),
        _query_recall_values(rows, scores),
    )


def _fit_train_audit_matrix(
    variants: tuple[tuple[str, Path], ...],
    event_history_path: Path,
    specs,
    selected_features: tuple[str, ...],
) -> dict[str, Any]:
    """Fit on each negative pool and score every model on every audit pool."""
    matrices = []
    common_cutoffs = None
    for label, candidate_path in variants:
        rows, cutoffs, feature_seconds = _build_matrix(
            candidate_path,
            event_history_path,
            specs,
            (0.50, 0.70, 0.85),
            common_cutoffs[2] if common_cutoffs is not None else None,
        )
        if common_cutoffs is None:
            common_cutoffs = cutoffs
        elif cutoffs != common_cutoffs:
            raise DataContractError(
                f"Temporal split changed for train/audit variant {label}."
            )
        for column in CATEGORICAL_FEATURES:
            rows[column] = rows[column].fillna("__MISSING__").astype(str)
        rows = rows.sort_values(
            ["event_time", "query_id", "label", "item_id"],
            ascending=[True, True, False, True],
            kind="stable",
        ).reset_index(drop=True)
        train_mask, selection_mask, audit_mask = _masks(rows, cutoffs)
        matrices.append(
            {
                "label": label,
                "input": str(candidate_path),
                "rows": rows,
                "train_mask": train_mask,
                "selection_mask": selection_mask,
                "audit_mask": audit_mask,
                "feature_generation_seconds": feature_seconds,
            }
        )

    reference = matrices[0]
    reference_train_rows = int(reference["train_mask"].sum())
    reference_selection_rows = int(reference["selection_mask"].sum())
    reference_audit_rows = reference["rows"].loc[reference["audit_mask"]]
    reference_query_ids = set(reference_audit_rows["query_id"].astype(str).unique())
    reference_positive_signature = _audit_positive_signature(reference_audit_rows)
    reference_audit_row_count = len(reference_audit_rows)
    for matrix in matrices[1:]:
        if int(matrix["train_mask"].sum()) != reference_train_rows or int(
            matrix["selection_mask"].sum()
        ) != reference_selection_rows:
            raise DataContractError(
                f"Train/audit variant {matrix['label']} changed train or selection row counts."
            )
        audit_rows = matrix["rows"].loc[matrix["audit_mask"]]
        if set(audit_rows["query_id"].astype(str).unique()) != reference_query_ids:
            raise DataContractError(
                f"Train/audit variant {matrix['label']} changed audit query IDs."
            )
        if _audit_positive_signature(audit_rows) != reference_positive_signature:
            raise DataContractError(
                f"Train/audit variant {matrix['label']} changed positive candidates."
            )
        if len(audit_rows) != reference_audit_row_count:
            raise DataContractError(
                f"Train/audit variant {matrix['label']} changed audit candidate count."
            )

    cells: dict[tuple[str, str], dict[str, Any]] = {}
    training_runs = []
    for train_matrix in matrices:
        rows = train_matrix["rows"]
        train_mask = train_matrix["train_mask"]
        selection_mask = train_matrix["selection_mask"]
        audit_mask = train_matrix["audit_mask"]
        train_and_selection = train_mask | selection_mask
        _, manual_selection = _fit_and_score(
            rows, MANUAL_FEATURE_NAMES, train_mask, selection_mask
        )
        _, automatic_selection = _fit_and_score(
            rows, selected_features, train_mask, selection_mask
        )
        manual_model, _ = _fit_and_score(
            rows,
            MANUAL_FEATURE_NAMES,
            train_and_selection,
            audit_mask,
            manual_selection["iterations"],
        )
        automatic_model, _ = _fit_and_score(
            rows,
            selected_features,
            train_and_selection,
            audit_mask,
            automatic_selection["iterations"],
        )
        model_specs = {
            "manual": {
                "model": manual_model,
                "features": MANUAL_FEATURE_NAMES,
                "selection": manual_selection,
            },
            "automatic": {
                "model": automatic_model,
                "features": selected_features,
                "selection": automatic_selection,
            },
        }
        training_runs.append(
            {
                "training_variant": train_matrix["label"],
                "training_input": train_matrix["input"],
                "train_rows": int(train_mask.sum()),
                "selection_rows": int(selection_mask.sum()),
                "training_cutoffs": [str(value) for value in common_cutoffs],
                "manual_selection": manual_selection,
                "automatic_selection": automatic_selection,
                "feature_generation_seconds": train_matrix[
                    "feature_generation_seconds"
                ],
            }
        )

        for audit_matrix in matrices:
            audit_rows = audit_matrix["rows"].loc[audit_matrix["audit_mask"]]
            cell_metrics = {}
            cell_query_metrics = {}
            for model_name, model_data in model_specs.items():
                current_metrics, current_ndcg, current_recall = _evaluate_model(
                    model_data["model"], audit_rows, model_data["features"]
                )
                cell_metrics[model_name] = current_metrics
                cell_query_metrics[model_name] = {
                    "ndcg@3": current_ndcg,
                    "recall@3": current_recall,
                }
            paired = _paired_bootstrap(
                cell_query_metrics["manual"]["ndcg@3"],
                cell_query_metrics["automatic"]["ndcg@3"],
                metric_name="automatic_minus_manual_ndcg@3",
            )
            cells[(train_matrix["label"], audit_matrix["label"])] = {
                "train_variant": train_matrix["label"],
                "audit_variant": audit_matrix["label"],
                "manual_metrics": cell_metrics["manual"],
                "automatic_metrics": cell_metrics["automatic"],
                "paired_automatic_minus_manual": paired,
                "query_metrics": cell_query_metrics,
            }

    training_reference = matrices[0]["label"]
    audit_reference = matrices[0]["label"]

    def paired_metric_report(
        reference_cell: dict[str, Any],
        comparison_cell: dict[str, Any],
        metric_prefix: str,
    ) -> dict[str, Any]:
        result = {}
        for model_name in ("manual", "automatic"):
            result[model_name] = {}
            for metric_name in ("ndcg@3", "recall@3"):
                result[model_name][metric_name] = _paired_bootstrap(
                    reference_cell["query_metrics"][model_name][metric_name],
                    comparison_cell["query_metrics"][model_name][metric_name],
                    metric_name=f"{metric_prefix}_{metric_name}",
                )
        return result

    training_seed_effects = []
    for audit_matrix in matrices:
        for train_matrix in matrices[1:]:
            training_seed_effects.append(
                {
                    "fixed_audit_variant": audit_matrix["label"],
                    "reference_training_variant": training_reference,
                    "comparison_training_variant": train_matrix["label"],
                    "paired_metrics": paired_metric_report(
                        cells[(training_reference, audit_matrix["label"])],
                        cells[(train_matrix["label"], audit_matrix["label"])],
                        "training_pool_change",
                    ),
                }
            )

    audit_seed_effects = []
    for train_matrix in matrices:
        for audit_matrix in matrices[1:]:
            audit_seed_effects.append(
                {
                    "fixed_training_variant": train_matrix["label"],
                    "reference_audit_variant": audit_reference,
                    "comparison_audit_variant": audit_matrix["label"],
                    "paired_metrics": paired_metric_report(
                        cells[(train_matrix["label"], audit_reference)],
                        cells[(train_matrix["label"], audit_matrix["label"])],
                        "audit_pool_change",
                    ),
                }
            )

    interactions = []
    for train_matrix in matrices[1:]:
        for audit_matrix in matrices[1:]:
            interaction = {
                "training_comparison": train_matrix["label"],
                "training_reference": training_reference,
                "audit_comparison": audit_matrix["label"],
                "audit_reference": audit_reference,
                "metrics": {},
            }
            for model_name in ("manual", "automatic"):
                interaction["metrics"][model_name] = {}
                for metric_name in ("ndcg@3", "recall@3"):
                    # Difference in training-pool effects between the comparison
                    # audit pool and the reference audit pool.
                    train_audit = cells[(train_matrix["label"], audit_matrix["label"])][
                        "query_metrics"
                    ][model_name][metric_name]
                    reference_reference = cells[(training_reference, audit_reference)][
                        "query_metrics"
                    ][model_name][metric_name]
                    reference_train = cells[(training_reference, audit_matrix["label"])][
                        "query_metrics"
                    ][model_name][metric_name]
                    train_reference = cells[(train_matrix["label"], audit_reference)][
                        "query_metrics"
                    ][model_name][metric_name]
                    positive = {
                        key: train_audit[key] + reference_reference[key]
                        for key in train_audit.keys()
                        & reference_reference.keys()
                        & reference_train.keys()
                        & train_reference.keys()
                    }
                    negative = {
                        key: reference_train[key] + train_reference[key]
                        for key in positive
                    }
                    interaction["metrics"][model_name][metric_name] = _paired_bootstrap(
                        negative,
                        positive,
                        metric_name="difference_in_training_effects",
                    )
            interactions.append(interaction)

    diagonal_samples = []
    for matrix in matrices:
        cell = cells[(matrix["label"], matrix["label"])]
        diagonal_samples.append(
            {
                "label": matrix["label"],
                "input": matrix["input"],
                "cutoffs": [str(value) for value in common_cutoffs],
                "rows": {
                    "train": int(matrix["train_mask"].sum()),
                    "selection": int(matrix["selection_mask"].sum()),
                    "audit": int(matrix["audit_mask"].sum()),
                },
                "audit_query_count": len(reference_query_ids),
                "feature_generation_seconds": matrix[
                    "feature_generation_seconds"
                ],
                "manual": {
                    "selection": next(
                        run["manual_selection"]
                        for run in training_runs
                        if run["training_variant"] == matrix["label"]
                    ),
                    "audit": cell["manual_metrics"],
                },
                "automatic": {
                    "selection": next(
                        run["automatic_selection"]
                        for run in training_runs
                        if run["training_variant"] == matrix["label"]
                    ),
                    "audit": cell["automatic_metrics"],
                },
                "paired_automatic_minus_manual": cell[
                    "paired_automatic_minus_manual"
                ],
                "audit_query_ndcg": {
                    "manual": cell["query_metrics"]["manual"]["ndcg@3"],
                    "automatic": cell["query_metrics"]["automatic"]["ndcg@3"],
                },
            }
        )

    serialized_cells = []
    for cell in cells.values():
        serialized_cells.append(
            {
                key: value
                for key, value in cell.items()
                if key != "query_metrics"
            }
        )
    return {
        "training_variants": [matrix["label"] for matrix in matrices],
        "audit_variants": [matrix["label"] for matrix in matrices],
        "training_cutoffs": [str(value) for value in common_cutoffs],
        "same_audit_query_ids": True,
        "same_audit_positive_candidates": True,
        "same_audit_candidate_count": reference_audit_row_count,
        "audit_query_count": len(reference_query_ids),
        "training_runs": training_runs,
        "cells": serialized_cells,
        "training_seed_effects": training_seed_effects,
        "audit_seed_effects": audit_seed_effects,
        "train_audit_interactions": interactions,
        "diagonal_samples": diagonal_samples,
        "_query_metrics_by_cell": cells,
    }


def _audit_positive_signature(rows: pd.DataFrame) -> tuple[tuple[Any, ...], ...]:
    positive_rows = rows.loc[
        rows["label"].astype(int) == 1,
        ["query_id", "user_id", "item_id", "event_time"],
    ].drop_duplicates()
    positive_rows = positive_rows.sort_values(
        ["query_id", "user_id", "item_id", "event_time"], kind="stable"
    )
    return tuple(positive_rows.itertuples(index=False, name=None))


def run_sampling_sensitivity(
    negative_variants: tuple[tuple[str, Path], ...],
    query_variants: tuple[tuple[str, Path], ...],
    exclusion_inputs: tuple[Path, ...],
    feature_summary_path: Path,
    event_history_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Compare negative-pool variation on fixed queries and disjoint query cohorts."""
    negative_group = _check_shared_sampling_config(
        negative_variants, "negative_sampling_seed"
    )
    query_group = _check_shared_sampling_config(query_variants, "query_seed")
    summary_path = feature_summary_path.expanduser().resolve()
    history_path = event_history_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    for path in (summary_path, history_path):
        if not path.is_file():
            raise DataContractError(f"Input file does not exist: {path}")
    search_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selected_features = tuple(search_summary.get("selected_features", ()))
    if not selected_features:
        raise DataContractError(
            "Feature summary has no selected_features from a frozen search."
        )
    specs_by_name = {
        spec.name: spec for spec in generate_feature_specs(("user", "item"))
    }
    required_features = {*MANUAL_FEATURE_NAMES, *selected_features}
    if missing := required_features - set(specs_by_name):
        raise DataContractError(
            f"Sampling check requires unknown feature formulas: {sorted(missing)}"
        )
    specs = tuple(specs_by_name[name] for name in sorted(required_features))

    exclusion_paths = tuple(path.expanduser().resolve() for path in exclusion_inputs)
    excluded_ids: set[str] = set()
    for path in exclusion_paths:
        excluded_ids.update(_query_ids(path))

    negative_query_sets = [_query_ids(path) for _, path in negative_variants]
    if any(query_ids != negative_query_sets[0] for query_ids in negative_query_sets[1:]):
        raise DataContractError(
            "Negative-sensitivity inputs must contain exactly the same query IDs."
        )
    positive_signatures = [_positive_signature(path) for _, path in negative_variants]
    if any(signature != positive_signatures[0] for signature in positive_signatures[1:]):
        raise DataContractError(
            "Negative-sensitivity inputs must have identical positive candidates."
        )
    negative_sets = [_negative_pairs(path) for _, path in negative_variants]
    if any(candidate_set == negative_sets[0] for candidate_set in negative_sets[1:]):
        raise DataContractError("At least two negative pools are identical.")

    query_sets = [_query_ids(path) for _, path in query_variants]
    query_frequency: dict[str, int] = {}
    for query_ids in query_sets:
        for query_id in query_ids:
            query_frequency[query_id] = query_frequency.get(query_id, 0) + 1
    overlapping_query_ids = {
        query_id for query_id, count in query_frequency.items() if count > 1
    }

    filtered_negative_paths: list[tuple[str, Path]] = []
    filtered_query_paths: list[tuple[str, Path]] = []
    filter_reports: dict[str, Any] = {}
    filter_root = output_dir / "filtered-candidates"
    for family_name, variants, extra_exclusions in (
        ("negative", negative_variants, set()),
        ("query", query_variants, overlapping_query_ids),
    ):
        for index, (label, input_path) in enumerate(variants):
            excluded = excluded_ids | extra_exclusions
            filtered_path = filter_root / f"{family_name}-{index}-{label}.parquet"
            report = _filter_queries(input_path, filtered_path, excluded)
            source_query_ids = _query_ids(input_path)
            report["excluded_query_overlap_with_other_query_cohorts"] = (
                len(source_query_ids & extra_exclusions)
            )
            report["excluded_query_overlap_with_prior_discovery_or_training"] = len(
                source_query_ids & excluded_ids
            )
            filter_reports[f"{family_name}:{label}"] = report
            if family_name == "negative":
                filtered_negative_paths.append((label, filtered_path))
            else:
                filtered_query_paths.append((label, filtered_path))

    filtered_negative_query_sets = [
        _query_ids(path) for _, path in filtered_negative_paths
    ]
    if any(
        query_ids != filtered_negative_query_sets[0]
        for query_ids in filtered_negative_query_sets[1:]
    ):
        raise DataContractError(
            "Filtering failed to preserve the common negative-sensitivity query cohort."
        )
    filtered_query_sets = [_query_ids(path) for _, path in filtered_query_paths]
    if any(
        first & second
        for index, first in enumerate(filtered_query_sets)
        for second in filtered_query_sets[index + 1 :]
    ):
        raise DataContractError("Filtered query-sensitivity cohorts still overlap.")

    started = time.perf_counter()
    negative_matrix = _fit_train_audit_matrix(
        tuple(filtered_negative_paths),
        history_path,
        specs,
        selected_features,
    )
    negative_samples = negative_matrix["diagonal_samples"]
    query_samples = [
        _fit_sample(label, path, history_path, specs, selected_features)
        for label, path in filtered_query_paths
    ]

    negative_reference = negative_samples[0]
    negative_comparisons = []
    reference_negatives = _negative_pairs(filtered_negative_paths[0][1])
    for sample, (_, variant_path) in zip(
        negative_samples[1:], filtered_negative_paths[1:]
    ):
        current_negatives = _negative_pairs(variant_path)
        intersection = len(reference_negatives & current_negatives)
        union = len(reference_negatives | current_negatives)
        negative_comparisons.append(
            {
                "reference": negative_reference["label"],
                "comparison": sample["label"],
                "negative_pair_jaccard": intersection / union if union else 1.0,
                "manual_query_ndcg": _paired_bootstrap(
                    negative_reference["audit_query_ndcg"]["manual"],
                    sample["audit_query_ndcg"]["manual"],
                ),
                "automatic_query_ndcg": _paired_bootstrap(
                    negative_reference["audit_query_ndcg"]["automatic"],
                    sample["audit_query_ndcg"]["automatic"],
                ),
            }
        )

    query_metrics = []
    for sample in query_samples:
        query_metrics.append(
            {
                "label": sample["label"],
                "audit_query_count": sample["audit_query_count"],
                "manual_ndcg@3": sample["manual"]["audit"]["metrics"]["ndcg@3"],
                "automatic_ndcg@3": sample["automatic"]["audit"]["metrics"]["ndcg@3"],
                "automatic_minus_manual_ndcg@3": sample[
                    "paired_automatic_minus_manual"
                ]["selected_minus_manual_ndcg@3"],
                "paired_automatic_minus_manual_interval": sample[
                    "paired_automatic_minus_manual"
                ]["bootstrap_95_percent_interval"],
            }
        )

    report = {
        "protocol": {
            "feature_search_summary": str(summary_path),
            "selected_features": list(selected_features),
            "manual_feature_count": len(MANUAL_FEATURE_NAMES),
            "model_seed": 42,
            "temporal_split": [0.50, 0.70, 0.85],
            "negative_sensitivity": (
                "same query IDs and identical positive rows; only the negative sampler seed varies"
            ),
            "query_sensitivity": (
                "query selection seed varies; negative sampler seed, catalog cutoffs, and sampling configuration stay fixed; overlapping query IDs are removed"
            ),
            "exclusion_inputs": [str(path) for path in exclusion_paths],
            "history_rule": "event_time strictly before candidate timestamp",
            "candidate_semantics": "sampled unknowns, not observed non-clicks",
        },
        "negative_sampling": {
            "sampling_config": negative_group,
            "filtering": {
                label: value
                for label, value in filter_reports.items()
                if label.startswith("negative:")
            },
            "samples": [_serialize_sample(sample) for sample in negative_samples],
            "paired_comparisons_to_first_pool": negative_comparisons,
            "train_audit_matrix": {
                key: value
                for key, value in negative_matrix.items()
                if not key.startswith("_") and key != "diagonal_samples"
            },
        },
        "query_sampling": {
            "sampling_config": query_group,
            "query_overlap_removed": len(overlapping_query_ids),
            "filtering": {
                label: value
                for label, value in filter_reports.items()
                if label.startswith("query:")
            },
            "samples": [_serialize_sample(sample) for sample in query_samples],
            "metrics_by_disjoint_cohort": query_metrics,
        },
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report

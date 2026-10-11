from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import catboost
import numpy as np
import pandas as pd
from catboost import Pool

from recsys_mvp.auto_features import FeatureSpec
from recsys_mvp.data import DataContractError
from recsys_mvp.evaluation import evaluate_predictions
from recsys_mvp.feature_search import (
    CATEGORICAL_FEATURES,
    MANUAL_FEATURE_NAMES,
    _build_matrix,
)
from recsys_mvp.robust_search import _masks
from recsys_mvp.sampling_sensitivity import (
    _ordered_group_rows,
    _paired_bootstrap,
    _query_ndcg_values,
)


PARAMETER_GRID = (
    {"depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 3.0},
    {"depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 3.0},
    {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3.0},
    {"depth": 10, "learning_rate": 0.05, "l2_leaf_reg": 3.0},
    {"depth": 8, "learning_rate": 0.03, "l2_leaf_reg": 3.0},
    {"depth": 8, "learning_rate": 0.10, "l2_leaf_reg": 3.0},
    {"depth": 8, "learning_rate": 0.15, "l2_leaf_reg": 3.0},
    {"depth": 8, "learning_rate": 0.05, "l2_leaf_reg": 10.0},
)
MAX_ITERATIONS = 500
EARLY_STOPPING_ROUNDS = 50
SPLIT_QUANTILES = (0.50, 0.70, 0.85)


def _manual_feature_specs() -> tuple[FeatureSpec, ...]:
    specs = []
    for entity in ("user", "item"):
        for days in (1, 7, 30, 90):
            specs.append(FeatureSpec(entity, "listen_plus", "count", days))
        aggregate = "distinct_items" if entity == "user" else "distinct_users"
        for days in (30, 90):
            specs.append(FeatureSpec(entity, "listen_plus", aggregate, days))
    names = set(MANUAL_FEATURE_NAMES)
    selected = tuple(spec for spec in specs if spec.name in names)
    if {spec.name for spec in selected} != names:
        raise ValueError("Manual feature definitions do not match the configured control.")
    return selected


def _load_matrix(candidate_path: Path, event_history_path: Path):
    rows, cutoffs, feature_seconds = _build_matrix(
        candidate_path,
        event_history_path,
        _manual_feature_specs(),
        SPLIT_QUANTILES,
    )
    for column in CATEGORICAL_FEATURES:
        rows[column] = rows[column].fillna("__MISSING__").astype(str)
    return rows, cutoffs, feature_seconds


def _pool(rows: pd.DataFrame):
    ordered = _ordered_group_rows(rows)
    pool = Pool(
        data=ordered[[*CATEGORICAL_FEATURES, *MANUAL_FEATURE_NAMES]],
        label=ordered["label"].astype(float),
        group_id=ordered["query_id"].astype(str),
        cat_features=list(CATEGORICAL_FEATURES),
    )
    return pool, ordered


def _make_model(parameters: dict[str, Any], iterations: int) -> catboost.CatBoostClassifier:
    return catboost.CatBoostClassifier(
        iterations=iterations,
        depth=parameters["depth"],
        learning_rate=parameters["learning_rate"],
        l2_leaf_reg=parameters["l2_leaf_reg"],
        loss_function="Logloss",
        eval_metric="NDCG:top=3;type=Base",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=4,
    )


def _fit_selection(
    rows: pd.DataFrame,
    train_mask: pd.Series,
    selection_mask: pd.Series,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    train_pool, _ = _pool(rows.loc[train_mask])
    selection_pool, selection_rows = _pool(rows.loc[selection_mask])
    model = _make_model(parameters, MAX_ITERATIONS)
    started = time.perf_counter()
    model.fit(
        train_pool,
        eval_set=selection_pool,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
    )
    fit_seconds = round(time.perf_counter() - started, 3)
    scores = model.predict_proba(selection_pool)[:, 1]
    metrics = evaluate_predictions(
        selection_rows,
        pd.Series(scores, index=selection_rows.index),
    )
    return {
        "parameters": parameters,
        "iterations": model.get_best_iteration() + 1,
        "fit_seconds": fit_seconds,
        "metrics": metrics,
    }


def _fit_audit(
    rows: pd.DataFrame,
    train_mask: pd.Series,
    selection_mask: pd.Series,
    audit_mask: pd.Series,
    parameters: dict[str, Any],
    iterations: int,
) -> tuple[dict[str, Any], dict[tuple[str, str], float]]:
    fit_mask = train_mask | selection_mask
    train_pool, _ = _pool(rows.loc[fit_mask])
    audit_pool, audit_rows = _pool(rows.loc[audit_mask])
    model = _make_model(parameters, iterations)
    started = time.perf_counter()
    model.fit(train_pool)
    fit_seconds = round(time.perf_counter() - started, 3)
    scores = model.predict_proba(audit_pool)[:, 1]
    metrics = evaluate_predictions(audit_rows, pd.Series(scores, index=audit_rows.index))
    return (
        {
            "iterations": iterations,
            "fit_seconds": fit_seconds,
            "rows": len(audit_rows),
            "metrics": metrics,
        },
        _query_ndcg_values(audit_rows, scores),
    )


def run_catboost_tuning(
    discovery_inputs: tuple[Path, ...],
    holdout_input: Path,
    event_history_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    discovery_inputs = tuple(path.expanduser().resolve() for path in discovery_inputs)
    holdout_input = holdout_input.expanduser().resolve()
    if len(discovery_inputs) < 2:
        raise DataContractError("At least two discovery candidate files are required.")
    if len(set(discovery_inputs)) != len(discovery_inputs):
        raise DataContractError("Discovery candidate files must be unique.")
    if holdout_input in discovery_inputs:
        raise DataContractError("Holdout input must not be used for parameter selection.")

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    discoveries = []
    for candidate_path in discovery_inputs:
        rows, cutoffs, feature_seconds = _load_matrix(candidate_path, event_history_path)
        train_mask, selection_mask, _ = _masks(rows, cutoffs)
        fold_results = []
        for parameters in PARAMETER_GRID:
            result = _fit_selection(rows, train_mask, selection_mask, parameters)
            fold_results.append(result)
            print(
                json.dumps(
                    {
                        "fold": candidate_path.stem,
                        "parameters": parameters,
                        "selection_ndcg@3": result["metrics"]["ndcg@3"],
                        "iterations": result["iterations"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        discoveries.append(
            {
                "input": str(candidate_path.expanduser().resolve()),
                "cutoffs": [str(value) for value in cutoffs],
                "rows": {
                    "train": int(train_mask.sum()),
                    "selection": int(selection_mask.sum()),
                },
                "feature_generation_seconds": feature_seconds,
                "configurations": fold_results,
            }
        )

    summary_by_config: list[dict[str, Any]] = []
    for index, parameters in enumerate(PARAMETER_GRID):
        fold_results = [fold["configurations"][index] for fold in discoveries]
        ndcgs = [float(result["metrics"]["ndcg@3"]) for result in fold_results]
        summary_by_config.append(
            {
                "parameters": parameters,
                "mean_selection_ndcg@3": float(np.mean(ndcgs)),
                "std_selection_ndcg@3": float(np.std(ndcgs, ddof=0)),
                "mean_selection_recall@3": float(
                    np.mean([result["metrics"]["recall@3"] for result in fold_results])
                ),
                "iterations_by_fold": [result["iterations"] for result in fold_results],
                "median_iterations": int(
                    round(statistics.median(result["iterations"] for result in fold_results))
                ),
                "folds": len(fold_results),
            }
        )
    summary_by_config.sort(
        key=lambda result: (
            -result["mean_selection_ndcg@3"],
            result["std_selection_ndcg@3"],
            result["parameters"]["depth"],
        )
    )
    chosen = summary_by_config[0]
    default_config = next(
        item for item in summary_by_config if item["parameters"] == PARAMETER_GRID[0]
    )

    holdout_rows, holdout_cutoffs, holdout_feature_seconds = _load_matrix(holdout_input, event_history_path)
    train_mask, selection_mask, audit_mask = _masks(holdout_rows, holdout_cutoffs)
    default_audit, default_ndcg = _fit_audit(
        holdout_rows,
        train_mask,
        selection_mask,
        audit_mask,
        default_config["parameters"],
        default_config["median_iterations"],
    )
    tuned_audit, tuned_ndcg = _fit_audit(
        holdout_rows,
        train_mask,
        selection_mask,
        audit_mask,
        chosen["parameters"],
        chosen["median_iterations"],
    )
    paired = _paired_bootstrap(
        default_ndcg,
        tuned_ndcg,
        metric_name="tuned_minus_default_ndcg@3",
    )
    report = {
        "protocol": {
            "candidate_sampling": "same mixed candidate protocol for all folds; unknown negatives, not logged impressions",
            "selection_metric": "NDCG@3 across temporal selection partitions; mean across discovery cohorts",
            "audit_split": "holdout candidate cohort, 70th-85th timestamp percentiles; not used for tuning",
            "features": list(MANUAL_FEATURE_NAMES),
            "split_quantiles": list(SPLIT_QUANTILES),
            "max_iterations": MAX_ITERATIONS,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "grid": list(PARAMETER_GRID),
            "catboost_version": catboost.__version__,
        },
        "discovery": discoveries,
        "ranking": summary_by_config,
        "chosen": chosen,
        "default_control": default_config,
        "holdout": {
            "input": str(holdout_input.expanduser().resolve()),
            "cutoffs": [str(value) for value in holdout_cutoffs],
            "rows": {
                "train": int(train_mask.sum()),
                "selection": int(selection_mask.sum()),
                "audit": int(audit_mask.sum()),
            },
            "queries": int(holdout_rows.loc[audit_mask, "query_id"].nunique()),
            "feature_generation_seconds": holdout_feature_seconds,
            "default": default_audit,
            "tuned": tuned_audit,
            "paired_comparison": paired,
        },
    }
    output_path = output_dir / "summary.json"
    report["output"] = str(output_path)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return report

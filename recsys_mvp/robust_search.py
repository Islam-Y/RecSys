from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import catboost
import numpy as np
import pandas as pd

from recsys_mvp.auto_features import (
    SEARCH_WINDOWS_DAYS,
    FeatureSpec,
    generate_feature_specs,
)
from recsys_mvp.data import DataContractError
from recsys_mvp.evaluation import evaluate_ranking_scores
from recsys_mvp.feature_search import (
    CATEGORICAL_FEATURES,
    MANUAL_FEATURE_NAMES,
    _build_matrix,
    _fit_and_score,
    _paired_ndcg_delta,
)


DISCOVERY_SPLITS = ((0.35, 0.55, 0.70), (0.50, 0.70, 0.85))
ROBUST_SELECTION_SCORE = "mean_ndcg@3 - 0.5 * population_stddev(ndcg@3)"


def _rank_features(
    rows: pd.DataFrame,
    specs: tuple[FeatureSpec, ...],
    train_mask: pd.Series,
    selection_mask: pd.Series,
    feature_budget: int,
) -> tuple[list[str], dict[str, Any]]:
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
    model, fit = _fit_and_score(
        rows, numeric_pool, train_mask, selection_mask
    )
    names = [*CATEGORICAL_FEATURES, *numeric_pool]
    importances = dict(
        zip(names, map(float, model.get_feature_importance()))
    )
    ranked = sorted(numeric_pool, key=lambda name: (-importances[name], name))
    return ranked, {
        "nonconstant_formula_count": len(numeric_pool),
        "importance": importances,
        "full_pool_selection": fit,
    }


def _masks(rows: pd.DataFrame, cutoffs: tuple[Any, Any, Any]):
    train = rows["event_time"] <= cutoffs[0]
    selection = (rows["event_time"] > cutoffs[0]) & (
        rows["event_time"] <= cutoffs[1]
    )
    audit = (rows["event_time"] > cutoffs[1]) & (rows["event_time"] <= cutoffs[2])
    if not train.any() or not selection.any() or not audit.any():
        raise DataContractError("Temporal feature-search split has an empty partition.")
    if rows.loc[train, "label"].nunique() < 2:
        raise DataContractError("Training partition must contain both labels.")
    return train, selection, audit


def run_robust_feature_search(
    discovery_inputs: tuple[Path, ...],
    holdout_input: Path,
    event_history_path: Path,
    output_dir: Path,
    entities: tuple[str, ...] = ("user", "item"),
    windows_days: tuple[int, ...] = SEARCH_WINDOWS_DAYS,
    feature_budget: int = 12,
) -> dict[str, Any]:
    """Select formulas across discovery folds and evaluate once on a fresh candidate sample."""
    if not discovery_inputs:
        raise DataContractError("At least one discovery candidate file is required.")
    if feature_budget < len(MANUAL_FEATURE_NAMES):
        raise DataContractError(
            f"Feature budget must be at least {len(MANUAL_FEATURE_NAMES)} "
            "to compare with the fixed manual control."
        )
    discovery_paths = tuple(path.expanduser().resolve() for path in discovery_inputs)
    holdout_path = holdout_input.expanduser().resolve()
    history_path = event_history_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if holdout_path in discovery_paths:
        raise DataContractError("Holdout input must not be included in discovery inputs.")
    for path in (*discovery_paths, holdout_path, history_path):
        if not path.is_file():
            raise DataContractError(f"Input file does not exist: {path}")

    specs = generate_feature_specs(entities, windows_days)
    spec_by_name = {spec.name: spec for spec in specs}
    missing_manual = set(MANUAL_FEATURE_NAMES) - set(spec_by_name)
    if missing_manual:
        raise DataContractError(
            "The search space must include the manual control features: "
            f"{sorted(missing_manual)}"
        )
    candidate_budgets = sorted({4, 8, feature_budget} & set(range(1, feature_budget + 1)))
    fold_rows: list[dict[str, Any]] = []
    fold_rankings: list[list[str]] = []
    started = time.perf_counter()

    for candidate_path in discovery_paths:
        for quantiles in DISCOVERY_SPLITS:
            rows, cutoffs, feature_seconds = _build_matrix(
                candidate_path, history_path, specs, quantiles
            )
            for column in CATEGORICAL_FEATURES:
                rows[column] = rows[column].fillna("__MISSING__").astype(str)
            train_mask, selection_mask, _ = _masks(rows, cutoffs)
            ranked, ranking_report = _rank_features(
                rows, specs, train_mask, selection_mask, feature_budget
            )
            fold_rankings.append(ranked)
            manual_model, manual_selection = _fit_and_score(
                rows, MANUAL_FEATURE_NAMES, train_mask, selection_mask
            )
            del manual_model
            fold_rows.append(
                {
                    "candidate_input": str(candidate_path),
                    "split_quantiles": list(quantiles),
                    "cutoffs": [str(value) for value in cutoffs],
                    "rows": {
                        "train": int(train_mask.sum()),
                        "selection": int(selection_mask.sum()),
                    },
                    "feature_generation_seconds": feature_seconds,
                    "ranking": ranking_report,
                    "manual_selection": manual_selection,
                    "ranked_features": ranked,
                }
            )

    rank_positions: dict[str, list[float]] = {spec.name: [] for spec in specs}
    for ranked in fold_rankings:
        rank_map = {name: index / max(1, len(ranked) - 1) for index, name in enumerate(ranked)}
        for name in rank_positions:
            rank_positions[name].append(rank_map.get(name, 1.0))
    mean_rank = {
        name: statistics.fmean(positions) for name, positions in rank_positions.items()
    }
    consensus_ranked = sorted(rank_positions, key=lambda name: (mean_rank[name], name))

    selection_scores: list[dict[str, Any]] = []
    chosen_budget: int | None = None
    chosen_key: tuple[float, float, int] | None = None
    for budget in candidate_budgets:
        fold_metrics = []
        for candidate_path, quantiles in (
            (path, split)
            for path in discovery_paths
            for split in DISCOVERY_SPLITS
        ):
            rows, cutoffs, _ = _build_matrix(
                candidate_path, history_path, specs, quantiles
            )
            for column in CATEGORICAL_FEATURES:
                rows[column] = rows[column].fillna("__MISSING__").astype(str)
            train_mask, selection_mask, _ = _masks(rows, cutoffs)
            _, result = _fit_and_score(
                rows, tuple(consensus_ranked[:budget]), train_mask, selection_mask
            )
            fold_metrics.append(
                {
                    "candidate_input": str(candidate_path),
                    "split_quantiles": list(quantiles),
                    **result,
                }
            )
        ndcgs = [row["metrics"]["ndcg@3"] for row in fold_metrics]
        recalls = [row["metrics"]["recall@3"] for row in fold_metrics]
        average_ndcg = statistics.fmean(ndcgs)
        std_ndcg = statistics.pstdev(ndcgs)
        robust_score = average_ndcg - 0.5 * std_ndcg
        key = (robust_score, statistics.fmean(recalls), -budget)
        if chosen_key is None or key > chosen_key:
            chosen_key = key
            chosen_budget = budget
        selection_scores.append(
            {
                "feature_budget": budget,
                "mean_ndcg@3": average_ndcg,
                "population_stddev_ndcg@3": std_ndcg,
                "robust_selection_score": robust_score,
                "mean_recall@3": statistics.fmean(recalls),
                "folds": fold_metrics,
            }
        )
    if chosen_budget is None:
        raise DataContractError("No feature budget was evaluated.")
    selected_names = tuple(consensus_ranked[:chosen_budget])

    holdout_rows, holdout_cutoffs, feature_seconds = _build_matrix(
        holdout_path, history_path, specs, (0.50, 0.70, 0.85)
    )
    for column in CATEGORICAL_FEATURES:
        holdout_rows[column] = holdout_rows[column].fillna("__MISSING__").astype(str)
    train_mask, selection_mask, audit_mask = _masks(holdout_rows, holdout_cutoffs)
    training_mask = train_mask | selection_mask
    _, manual_selection = _fit_and_score(
        holdout_rows, MANUAL_FEATURE_NAMES, train_mask, selection_mask
    )
    _, auto_selection = _fit_and_score(
        holdout_rows, selected_names, train_mask, selection_mask
    )
    manual_model, manual_audit = _fit_and_score(
        holdout_rows,
        MANUAL_FEATURE_NAMES,
        training_mask,
        audit_mask,
        manual_selection["iterations"],
    )
    auto_model, auto_audit = _fit_and_score(
        holdout_rows,
        selected_names,
        training_mask,
        audit_mask,
        auto_selection["iterations"],
    )
    audit_rows = holdout_rows.loc[audit_mask]
    manual_scores = manual_model.predict_proba(
        audit_rows[[*CATEGORICAL_FEATURES, *MANUAL_FEATURE_NAMES]]
    )[:, 1]
    auto_scores = auto_model.predict_proba(
        audit_rows[[*CATEGORICAL_FEATURES, *selected_names]]
    )[:, 1]
    paired_delta = _paired_ndcg_delta(audit_rows, manual_scores, auto_scores)
    popularity = evaluate_ranking_scores(
        audit_rows,
        audit_rows["label"].astype(int),
        audit_rows["item_listen_plus_count_30d"].astype(float),
        k=3,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manual_model.save_model(str(output_dir / "manual.cbm"))
    auto_model.save_model(str(output_dir / "robust-selected.cbm"))
    report = {
        "protocol": {
            "discovery_inputs": [str(path) for path in discovery_paths],
            "holdout_input": str(holdout_path),
            "event_history_input": str(history_path),
            "discovery_splits": [list(split) for split in DISCOVERY_SPLITS],
            "holdout_split": [0.50, 0.70, 0.85],
            "selection_score": ROBUST_SELECTION_SCORE,
            "consensus_rule": "mean normalized feature rank across discovery folds",
            "holdout_used_for_selection": False,
            "history_rule": "event_time strictly before candidate timestamp",
            "candidate_semantics": "sampled unknowns, not observed non-clicks",
            "model": "CatBoostClassifier depth=8, learning_rate=0.05, seed=42",
        },
        "generated_formulas": [spec.as_dict() for spec in specs],
        "consensus_ranking": [
            {"feature": name, "mean_normalized_rank": mean_rank[name]}
            for name in consensus_ranked
        ],
        "discovery_folds": fold_rows,
        "selection_scores": selection_scores,
        "chosen_feature_budget": chosen_budget,
        "selected_features": list(selected_names),
        "holdout": {
            "cutoffs": [str(value) for value in holdout_cutoffs],
            "rows": {
                "train": int(train_mask.sum()),
                "selection": int(selection_mask.sum()),
                "audit": int(audit_mask.sum()),
            },
            "feature_generation_seconds": feature_seconds,
            "manual": {
                "selection": manual_selection,
                "audit": manual_audit,
            },
            "automatic": {
                "selection": auto_selection,
                "audit": auto_audit,
            },
            "item_popularity_30d_audit": popularity,
            "paired_audit_comparison": paired_delta,
        },
        "runtime": {"total_seconds": round(time.perf_counter() - started, 3)},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report

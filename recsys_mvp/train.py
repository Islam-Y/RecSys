from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import catboost
import duckdb
import pandas as pd

from recsys_mvp.data import (
    DataContractError,
    create_examples_view,
    register_source,
)
from recsys_mvp.evaluation import evaluate_predictions, evaluate_ranking_scores
from recsys_mvp.features import WINDOWS_DAYS, build_feature_table


def _json_default(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _metrics_for_split(frame: pd.DataFrame, scores: pd.Series) -> dict[str, Any]:
    result = evaluate_predictions(frame, scores)
    result["rows"] = len(frame)
    result["positive_rate"] = float(frame["label"].mean()) if len(frame) else None
    return result


def _peak_process_memory_mb() -> float:
    peak_value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return round(peak_value / divisor, 1)


def run_baseline(
    input_path: Path,
    output_dir: Path,
    user_column: str = "user_id",
    item_column: str = "item_id",
    timestamp_column: str = "timestamp",
    label_column: str = "label",
    query_column: str = "query_id",
    history_input_path: Path | None = None,
    windows_days: tuple[int, ...] = WINDOWS_DAYS,
    include_trends: bool = False,
    include_recency: bool = False,
    evaluate_test: bool = True,
    train_quantile: float = 0.70,
    validation_quantile: float = 0.85,
    entity_windows_days: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, Any]:
    if not 0 < train_quantile < validation_quantile < 1:
        raise DataContractError(
            "Split quantiles must satisfy 0 < train < validation < 1."
        )
    run_started = time.perf_counter()
    connection = duckdb.connect()
    try:
        columns = register_source(connection, input_path)
        create_examples_view(
            connection,
            columns,
            user_column,
            item_column,
            timestamp_column,
            label_column,
            query_column,
        )

        total_rows, train_cutoff, validation_cutoff = connection.execute(
            f"""
            SELECT
                COUNT(*),
                quantile_disc(event_time, {train_quantile}),
                quantile_disc(event_time, {validation_quantile})
            FROM examples
            """
        ).fetchone()
        if total_rows < 3 or train_cutoff is None or validation_cutoff is None:
            raise DataContractError("Not enough rows to create temporal train/validation/test splits.")

        split_sizes = connection.execute(
            """
            SELECT
                COUNT(*) FILTER (WHERE event_time <= ?),
                COUNT(*) FILTER (WHERE event_time > ? AND event_time <= ?),
                COUNT(*) FILTER (WHERE event_time > ?)
            FROM examples
            """,
            [train_cutoff, train_cutoff, validation_cutoff, validation_cutoff],
        ).fetchone()
        if min(split_sizes) == 0:
            raise DataContractError(
                "Temporal split produced an empty partition. This often happens "
                "when many rows share the same timestamp; use coarser date-based "
                "cutoffs after inspecting the data."
            )

        if not evaluate_test:
            connection.execute(
                """
                CREATE TEMP TABLE evaluation_examples AS
                SELECT * FROM examples
                WHERE event_time <= ?
                """,
                [validation_cutoff],
            )
            connection.execute("DROP VIEW examples")
            connection.execute(
                """
                CREATE TEMP VIEW examples AS
                SELECT * FROM evaluation_examples
                """
            )

        feature_started = time.perf_counter()
        build_feature_table(
            connection,
            history_input_path,
            windows_days,
            include_trends,
            include_recency,
            entity_windows_days,
        )
        feature_generation_seconds = time.perf_counter() - feature_started

        all_rows = connection.execute("SELECT * FROM model_data ORDER BY row_id").fetchdf()
    finally:
        connection.close()

    all_rows["split"] = "test"
    all_rows.loc[all_rows["event_time"] <= train_cutoff, "split"] = "train"
    all_rows.loc[
        (all_rows["event_time"] > train_cutoff)
        & (all_rows["event_time"] <= validation_cutoff),
        "split",
    ] = "validation"

    feature_columns = [
        column
        for column in all_rows.columns
        if column
        not in {
            "row_id",
            "query_id",
            "event_time",
            "label",
            "split",
            "item_popularity_30d",
        }
    ]
    categorical_columns = ["user_id", "item_id"]
    for column in categorical_columns:
        all_rows[column] = all_rows[column].fillna("__MISSING__").astype(str)

    train_rows = all_rows.loc[all_rows["split"] == "train"]
    validation_rows = all_rows.loc[all_rows["split"] == "validation"]
    if train_rows["label"].nunique() < 2:
        raise DataContractError("The training partition must contain both labels 0 and 1.")

    model = catboost.CatBoostClassifier(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    training_started = time.perf_counter()
    model.fit(
        train_rows[feature_columns],
        train_rows["label"].astype(int),
        cat_features=categorical_columns,
        eval_set=(
            validation_rows[feature_columns],
            validation_rows["label"].astype(int),
        ),
        early_stopping_rounds=50,
        use_best_model=True,
    )
    training_seconds = time.perf_counter() - training_started

    output_path = output_dir.expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "catboost.cbm"
    model.save_model(str(model_path))

    metrics: dict[str, Any] = {
        "input": str(input_path.expanduser().resolve()),
        "source_columns": {
            "user": user_column,
            "item": item_column,
            "timestamp": timestamp_column,
            "label": label_column,
            "query": query_column,
        },
        "history_input": (
            str(history_input_path.expanduser().resolve())
            if history_input_path is not None
            else None
        ),
        "split": {
            "strategy": (
                "chronological global "
                f"{train_quantile:.0%}/"
                f"{validation_quantile - train_quantile:.0%}/"
                f"{1 - validation_quantile:.0%} by timestamp quantiles"
            ),
            "train_quantile": train_quantile,
            "validation_quantile": validation_quantile,
            "train_cutoff_inclusive": train_cutoff,
            "validation_cutoff_inclusive": validation_cutoff,
            "rows": {
                "train": split_sizes[0],
                "validation": split_sizes[1],
                "test": split_sizes[2],
            },
            "history_rule": "positive interactions with event_time strictly before the candidate timestamp",
            "timestamp_note": (
                "Yambda timestamps are relative seconds with 5-second granularity; "
                "the rendered 1970-based timestamps are an arbitrary origin, "
                "not calendar dates."
                if history_input_path is not None
                else None
            ),
        },
        "features": feature_columns,
        "feature_config": {
            "windows_days": list(sorted(set(windows_days))),
            "entity_windows_days": {
                prefix: list(
                    sorted(set((entity_windows_days or {}).get(prefix, windows_days)))
                )
                for prefix in ("user", "item", "pair")
            },
            "include_7d_trend_features": include_trends,
            "include_days_since_last_listen_features": include_recency,
        },
        "categorical_features": categorical_columns,
        "model": {
            "name": "CatBoostClassifier",
            "best_iteration": model.get_best_iteration(),
            "parameters": {
                "iterations": 500,
                "depth": 8,
                "learning_rate": 0.05,
                "random_seed": 42,
            },
        },
        "metrics": {},
        "model_file": str(model_path),
    }

    splits_to_evaluate = [
        ("train", train_rows),
        ("validation", validation_rows),
    ]
    if evaluate_test:
        splits_to_evaluate.append(
            ("test", all_rows.loc[all_rows["split"] == "test"])
        )

    scoring_started = time.perf_counter()
    for split_name, rows in splits_to_evaluate:
        probabilities = model.predict_proba(rows[feature_columns])[:, 1]
        metrics["metrics"][split_name] = _metrics_for_split(
            rows,
            pd.Series(probabilities, index=rows.index),
        )
        metrics["metrics"][split_name]["item_popularity_baseline"] = (
            evaluate_ranking_scores(
                rows,
                rows["label"].astype(int),
                rows["item_popularity_30d"].astype(float),
                k=3,
            )
        )

    metrics["experiment"] = {
        "test_metrics_evaluated": evaluate_test,
        "validation_is_used_for_early_stopping": True,
    }
    metrics["runtime"] = {
        "feature_generation_seconds": round(feature_generation_seconds, 3),
        "training_seconds": round(training_seconds, 3),
        "scoring_seconds": round(time.perf_counter() - scoring_started, 3),
        "end_to_end_seconds": round(time.perf_counter() - run_started, 3),
        "peak_process_memory_mb": _peak_process_memory_mb(),
    }

    metrics_path = output_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return metrics

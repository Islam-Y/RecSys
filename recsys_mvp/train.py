from __future__ import annotations

import json
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
from recsys_mvp.features import build_feature_table


def _json_default(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _metrics_for_split(frame: pd.DataFrame, scores: pd.Series) -> dict[str, Any]:
    result = evaluate_predictions(frame, scores)
    result["rows"] = len(frame)
    result["positive_rate"] = float(frame["label"].mean()) if len(frame) else None
    return result


def run_baseline(
    input_path: Path,
    output_dir: Path,
    user_column: str = "user_id",
    item_column: str = "item_id",
    timestamp_column: str = "timestamp",
    label_column: str = "label",
    query_column: str = "query_id",
    history_input_path: Path | None = None,
) -> dict[str, Any]:
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
        build_feature_table(connection, history_input_path)

        total_rows, train_cutoff, validation_cutoff = connection.execute(
            """
            SELECT
                COUNT(*),
                quantile_disc(event_time, 0.70),
                quantile_disc(event_time, 0.85)
            FROM model_data
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
            FROM model_data
            """,
            [train_cutoff, train_cutoff, validation_cutoff, validation_cutoff],
        ).fetchone()
        if min(split_sizes) == 0:
            raise DataContractError(
                "Temporal split produced an empty partition. This often happens "
                "when many rows share the same timestamp; use coarser date-based "
                "cutoffs after inspecting the data."
            )

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
        not in {"row_id", "query_id", "event_time", "label", "split"}
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

    output_path = output_dir.expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "catboost.cbm"
    model.save_model(str(model_path))

    metrics: dict[str, Any] = {
        "input": str(input_path.expanduser().resolve()),
        "history_input": (
            str(history_input_path.expanduser().resolve())
            if history_input_path is not None
            else None
        ),
        "split": {
            "strategy": "chronological global 70/15/15 by timestamp quantiles",
            "train_cutoff_inclusive": train_cutoff,
            "validation_cutoff_inclusive": validation_cutoff,
            "rows": {
                "train": split_sizes[0],
                "validation": split_sizes[1],
                "test": split_sizes[2],
            },
            "history_rule": "positive interactions with event_time strictly before the candidate timestamp",
            "timestamp_note": (
                "Yambda timestamps are relative 5-second bins; the rendered "
                "1970-based timestamps are an arbitrary origin, not calendar dates."
                if history_input_path is not None
                else None
            ),
        },
        "features": feature_columns,
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

    for split_name, rows in (
        ("train", train_rows),
        ("validation", validation_rows),
        ("test", all_rows.loc[all_rows["split"] == "test"]),
    ):
        probabilities = model.predict_proba(rows[feature_columns])[:, 1]
        metrics["metrics"][split_name] = _metrics_for_split(
            rows,
            pd.Series(probabilities, index=rows.index),
        )
        metrics["metrics"][split_name]["item_popularity_baseline"] = (
            evaluate_ranking_scores(
                rows,
                rows["label"].astype(int),
                rows["item_events_30d"].astype(float),
                k=3,
            )
        )

    metrics_path = output_path / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return metrics

from __future__ import annotations

import math
from typing import Any

import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


def evaluate_ranking_scores(
    frame: pd.DataFrame,
    labels: pd.Series,
    scores: pd.Series,
    k: int,
) -> dict[str, Any]:
    if frame["query_id"].notna().all():
        group_columns = ["user_id", "query_id"]
        groups = frame.groupby(group_columns, dropna=False, sort=False).indices.values()
    else:
        group_columns = ["user_id", "event_time"]
        groups = frame.groupby(group_columns, dropna=False, sort=False).indices.values()

    ndcg_values: list[float] = []
    recall_values: list[float] = []
    recommended_items: set[str] = set()
    evaluation_queries = 0
    evaluable_queries = 0

    labels_array = labels.to_numpy()
    scores_array = scores.to_numpy()
    item_array = frame["item_id"].to_numpy()

    for positions in groups:
        positions = list(positions)
        evaluation_queries += 1
        query_labels = labels_array[positions]
        query_scores = scores_array[positions]
        order = sorted(
            range(len(positions)),
            key=lambda index: (-query_scores[index], index),
        )
        top_indices = order[:k]
        recommended_items.update(str(item_array[positions[index]]) for index in top_indices)

        positive_count = int(query_labels.sum())
        if len(positions) < 2 or positive_count == 0:
            continue

        evaluable_queries += 1
        dcg = sum(
            float(query_labels[index]) / math.log2(rank + 2)
            for rank, index in enumerate(top_indices)
        )
        ideal_count = min(positive_count, k)
        idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_count))
        ndcg_values.append(dcg / idcg if idcg else 0.0)
        recall_values.append(
            float(query_labels[top_indices].sum()) / positive_count
        )

    catalog_size = frame["item_id"].nunique()
    return {
        f"ndcg@{k}": sum(ndcg_values) / len(ndcg_values) if ndcg_values else None,
        f"recall@{k}": sum(recall_values) / len(recall_values) if recall_values else None,
        "queries": evaluation_queries,
        "evaluable_queries": evaluable_queries,
        "mean_candidates_per_query": (
            float(len(frame) / evaluation_queries) if evaluation_queries else None
        ),
        f"catalog_coverage@{k}": (
            len(recommended_items) / catalog_size if catalog_size else None
        ),
        "query_grouping": (
            "user_id + query_id"
            if frame["query_id"].notna().any()
            else "user_id + event_time"
        ),
        "ranking_note": (
            None
            if evaluable_queries
            else "No multi-candidate query with a positive label; ranking metrics are unavailable."
        ),
    }


def evaluate_predictions(
    frame: pd.DataFrame,
    scores: pd.Series,
    k: int = 3,
) -> dict[str, Any]:
    labels = frame["label"].astype(int)
    result: dict[str, Any] = {}
    try:
        result["auc"] = float(roc_auc_score(labels, scores))
    except ValueError:
        result["auc"] = None
    result["log_loss"] = float(log_loss(labels, scores, labels=[0, 1]))
    result.update(evaluate_ranking_scores(frame, labels, scores, k))
    return result

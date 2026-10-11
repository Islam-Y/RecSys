from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from recsys_mvp.data import DataContractError


def _result_row(windows: tuple[int, ...], metrics: dict[str, Any]) -> dict[str, Any]:
    validation = metrics["metrics"]["validation"]
    runtime = metrics["runtime"]
    return {
        "windows_days": list(windows),
        "window_count": len(windows),
        "feature_count": len(metrics["features"]),
        "ndcg@3": validation["ndcg@3"],
        "recall@3": validation["recall@3"],
        "feature_generation_seconds": runtime["feature_generation_seconds"],
        "training_seconds": runtime["training_seconds"],
        "peak_process_memory_mb": runtime["peak_process_memory_mb"],
        "metrics_file": str(Path(metrics["model_file"]).with_name("metrics.json")),
    }


def _ranking_key(row: dict[str, Any]) -> tuple[float, float]:
    return row["ndcg@3"], row["recall@3"]


def run_window_search(
    input_path: Path,
    output_dir: Path,
    history_input_path: Path | None,
    base_windows: tuple[int, ...],
    candidate_windows: tuple[int, ...],
    user_column: str = "user_id",
    item_column: str = "item_id",
    timestamp_column: str = "timestamp",
    label_column: str = "label",
    query_column: str = "query_id",
    max_windows: int | None = None,
) -> dict[str, Any]:
    base_windows = tuple(sorted(set(base_windows)))
    candidate_windows = tuple(sorted(set(candidate_windows)))
    if not base_windows or not candidate_windows:
        raise DataContractError("Base and candidate windows must both be nonempty.")
    if set(base_windows) & set(candidate_windows):
        raise DataContractError("Candidate windows must not overlap base windows.")
    full_windows = tuple(sorted((*base_windows, *candidate_windows)))
    if max_windows is None:
        max_windows = len(full_windows)
    if not len(base_windows) <= max_windows <= len(full_windows):
        raise DataContractError(
            "Window budget must be between the base count and the full candidate count."
        )

    input_path = input_path.expanduser().resolve()
    history_input_path = (
        history_input_path.expanduser().resolve()
        if history_input_path is not None
        else None
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_files = [
        input_path,
        Path(__file__),
        Path(__file__).with_name("train.py"),
        Path(__file__).with_name("features.py"),
        Path(__file__).with_name("evaluation.py"),
        Path(__file__).with_name("data.py"),
    ]
    if history_input_path is not None:
        source_files.append(history_input_path)
    newest_source_mtime = max(path.stat().st_mtime for path in source_files)
    rows: dict[tuple[int, ...], dict[str, Any]] = {}
    source_columns = {
        "user": user_column,
        "item": item_column,
        "timestamp": timestamp_column,
        "label": label_column,
        "query": query_column,
    }

    def evaluate(windows: tuple[int, ...]) -> dict[str, Any]:
        windows = tuple(sorted(windows))
        if windows in rows:
            return rows[windows]

        run_dir = output_dir / ("windows-" + "-".join(map(str, windows)))
        metrics_path = run_dir / "metrics.json"
        metrics = None
        if metrics_path.is_file() and metrics_path.stat().st_mtime >= newest_source_mtime:
            cached = json.loads(metrics_path.read_text(encoding="utf-8"))
            if (
                cached.get("input") == str(input_path)
                and cached.get("source_columns") == source_columns
                and cached.get("history_input")
                == (str(history_input_path) if history_input_path is not None else None)
                and cached.get("feature_config", {}).get("windows_days") == list(windows)
                and cached.get("feature_config", {}).get("include_7d_trend_features") is False
                and cached.get("feature_config", {}).get("include_days_since_last_listen_features") is False
                and cached.get("experiment", {}).get("test_metrics_evaluated") is False
                and Path(cached.get("model_file", "")).is_file()
            ):
                metrics = cached

        if metrics is None:
            command = [
                sys.executable,
                "-m",
                "recsys_mvp",
                "baseline",
                str(input_path),
                "--output-dir",
                str(run_dir),
                "--windows-days",
                ",".join(map(str, windows)),
                "--validation-only",
                "--user-column",
                user_column,
                "--item-column",
                item_column,
                "--timestamp-column",
                timestamp_column,
                "--label-column",
                label_column,
                "--query-column",
                query_column,
            ]
            if history_input_path is not None:
                command.extend(("--history-input", str(history_input_path)))
            try:
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
            except subprocess.CalledProcessError as error:
                raise DataContractError(
                    f"Window configuration {windows} failed with exit code {error.returncode}."
                ) from error
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        row = _result_row(windows, metrics)
        rows[windows] = row
        print(
            f"windows={windows} ndcg@3={row['ndcg@3']:.4f} "
            f"recall@3={row['recall@3']:.4f} "
            f"features={row['feature_count']}",
            flush=True,
        )
        return row

    base = evaluate(base_windows)
    current = base_windows
    greedy_path = [base]
    remaining = set(candidate_windows)
    while remaining and len(current) < max_windows:
        candidates = [
            evaluate(tuple(sorted((*current, window))))
            for window in sorted(remaining)
        ]
        chosen = max(candidates, key=_ranking_key)
        greedy_path.append(chosen)
        next_windows = tuple(chosen["windows_days"])
        added_window = next(window for window in next_windows if window not in current)
        remaining.remove(added_window)
        current = next_windows

    full = evaluate(full_windows) if max_windows == len(full_windows) else None
    leave_one_out = (
        [
            {
                "removed_window_days": window,
                **evaluate(tuple(days for days in full_windows if days != window)),
            }
            for window in full_windows
        ]
        if full is not None
        else []
    )

    evaluated = sorted(rows.values(), key=lambda row: (row["window_count"], row["windows_days"]))
    best_by_budget = []
    for budget in range(len(base_windows), max_windows + 1):
        eligible = [row for row in evaluated if row["window_count"] <= budget]
        winner = max(eligible, key=_ranking_key)
        best_by_budget.append({"max_windows": budget, **winner})

    report = {
        "protocol": {
            "input": str(input_path),
            "history_input": str(history_input_path) if history_input_path else None,
            "base_windows_days": list(base_windows),
            "candidate_windows_days": list(candidate_windows),
            "max_windows": max_windows,
            "selection_metric": "validation ndcg@3, recall@3 as tie-breaker",
            "search": (
                "greedy forward additions from base; full-set leave-one-out "
                "only when the budget permits the full set"
            ),
            "test_metrics_evaluated": False,
            "selection_limit": "Best within each window budget among evaluated sets; not exhaustive",
        },
        "base": base,
        "full": full,
        "greedy_path": greedy_path,
        "leave_one_out": leave_one_out,
        "best_by_budget": best_by_budget,
        "evaluated": evaluated,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report

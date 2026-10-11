from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from recsys_mvp.data import DataContractError, profile_data
from recsys_mvp.yambda import prepare_yambda_candidates
from recsys_mvp.train import run_baseline
from recsys_mvp.features import WINDOWS_DAYS


def _parse_windows_days(value: str) -> tuple[int, ...]:
    try:
        windows = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Use comma-separated positive whole days, such as 1,7,30."
        ) from error
    if not windows or any(days < 1 for days in windows):
        raise argparse.ArgumentTypeError(
            "Feature windows must be positive whole days."
        )
    if len(set(windows)) != len(windows):
        raise argparse.ArgumentTypeError("Feature windows must not contain duplicates.")
    return tuple(sorted(windows))


def _add_column_options(parser: argparse.ArgumentParser, include_label: bool) -> None:
    parser.add_argument("--user-column", default="user_id")
    parser.add_argument("--item-column", default="item_id")
    parser.add_argument("--timestamp-column", default="timestamp")
    if include_label:
        parser.add_argument("--label-column", default="label")
    parser.add_argument("--query-column", default="query_id")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recsys-mvp",
        description="Audit interaction data and run a leakage-aware CatBoost baseline.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    profile = commands.add_parser("profile", help="Summarize a CSV, TSV, or Parquet table.")
    profile.add_argument("input", type=Path)
    _add_column_options(profile, include_label=True)

    baseline = commands.add_parser(
        "baseline",
        help="Train and evaluate a pointwise baseline from explicit binary candidates.",
    )
    baseline.add_argument("input", type=Path)
    baseline.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    baseline.add_argument(
        "--history-input",
        type=Path,
        help="Optional full positive-event history Parquet with user_id, item_id, event_time.",
    )
    baseline.add_argument(
        "--windows-days",
        type=_parse_windows_days,
        default=WINDOWS_DAYS,
        help="Comma-separated lookback windows in days; default: 1,7,30.",
    )
    baseline.add_argument(
        "--include-trends",
        action="store_true",
        help="Add previous-7-day counts and normalized 7-day activity changes.",
    )
    baseline.add_argument(
        "--include-recency",
        action="store_true",
        help="Add days since the last positive event within the largest window.",
    )
    baseline.add_argument(
        "--validation-only",
        action="store_true",
        help="Do not calculate or write test metrics; use validation for comparison.",
    )
    _add_column_options(baseline, include_label=True)

    yambda = commands.add_parser(
        "prepare-yambda",
        help="Create a sampled next-listen ranking dataset from Yambda-50M.",
    )
    yambda.add_argument("input", type=Path)
    yambda.add_argument(
        "--candidate-output",
        type=Path,
        default=Path("data/processed/yambda_candidates.parquet"),
    )
    yambda.add_argument(
        "--history-output",
        type=Path,
        default=Path("data/processed/yambda_history.parquet"),
    )
    yambda.add_argument("--query-sample-modulo", type=int, default=5000)
    yambda.add_argument("--negatives-per-query", type=int, default=20)
    yambda.add_argument(
        "--negative-strategy",
        choices=("uniform", "popular", "mixed"),
        default="mixed",
        help="Sample unknown items uniformly, from the warm-up popularity pool, or half-and-half.",
    )
    yambda.add_argument("--popular-pool-size", type=int, default=50000)
    yambda.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "profile":
            report = profile_data(
                args.input,
                args.user_column,
                args.item_column,
                args.timestamp_column,
                args.label_column,
                args.query_column,
            )
            print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
            return 0

        if args.command == "prepare-yambda":
            report = prepare_yambda_candidates(
                args.input,
                args.candidate_output,
                args.history_output,
                args.query_sample_modulo,
                args.negatives_per_query,
                args.seed,
                args.negative_strategy,
                args.popular_pool_size,
            )
            print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
            return 0

        metrics = run_baseline(
            args.input,
            args.output_dir,
            args.user_column,
            args.item_column,
            args.timestamp_column,
            args.label_column,
            args.query_column,
            args.history_input,
            args.windows_days,
            args.include_trends,
            args.include_recency,
            not args.validation_only,
        )
        print(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))
        return 0
    except (DataContractError, OSError) as error:
        parser.error(str(error))
    return 2

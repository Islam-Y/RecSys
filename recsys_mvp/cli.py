from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from recsys_mvp.auto_features import ENTITIES, SEARCH_WINDOWS_DAYS
from recsys_mvp.data import DataContractError, profile_data
from recsys_mvp.feature_search import run_feature_search
from recsys_mvp.features import WINDOWS_DAYS
from recsys_mvp.train import run_baseline
from recsys_mvp.window_search import run_window_search
from recsys_mvp.yambda import (
    prepare_cold_pair_candidates,
    prepare_unseen_pair_candidates,
    prepare_yambda_candidates,
    prepare_yambda_event_history,
)


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


def _parse_entities(value: str) -> tuple[str, ...]:
    entities = tuple(part.strip() for part in value.split(","))
    if (
        not entities
        or len(set(entities)) != len(entities)
        or set(entities) - set(ENTITIES)
    ):
        raise argparse.ArgumentTypeError("Use unique user,item,pair entity names.")
    return entities


def _parse_split_quantiles(value: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Use three increasing quantiles, such as 0.50,0.70,0.85."
        ) from error
    if len(values) != 3 or not 0 < values[0] < values[1] < values[2] < 1:
        raise argparse.ArgumentTypeError(
            "Use three increasing quantiles within 0-1."
        )
    return values


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
    for prefix in ("user", "item", "pair"):
        baseline.add_argument(
            f"--{prefix}-windows-days",
            type=_parse_windows_days,
            help=f"Override lookback windows for {prefix} features.",
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
    baseline.add_argument(
        "--train-quantile",
        type=float,
        default=0.70,
        help="Train cutoff as a global timestamp quantile; default: 0.70.",
    )
    baseline.add_argument(
        "--validation-quantile",
        type=float,
        default=0.85,
        help="Validation cutoff as a global timestamp quantile; default: 0.85.",
    )
    _add_column_options(baseline, include_label=True)

    search = commands.add_parser(
        "search-windows",
        help="Compare temporal window groups on validation under window-count budgets.",
    )
    search.add_argument("input", type=Path)
    search.add_argument("--output-dir", type=Path, default=Path("outputs/window-search"))
    search.add_argument("--history-input", type=Path)
    search.add_argument(
        "--base-windows",
        type=_parse_windows_days,
        default=WINDOWS_DAYS,
        help="Starting windows; default: 1,7,30.",
    )
    search.add_argument(
        "--candidate-windows",
        type=_parse_windows_days,
        default=(3, 14, 60, 90),
        help="Windows to try adding; default: 3,14,60,90.",
    )
    search.add_argument(
        "--max-windows",
        type=int,
        help="Maximum number of windows to evaluate; default: all candidates.",
    )
    _add_column_options(search, include_label=True)

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

    cold_pair = commands.add_parser(
        "cold-pair-control",
        help="Keep only queries whose positive items lack prior Listen+ pair history.",
    )
    cold_pair.add_argument("input", type=Path)
    cold_pair.add_argument("--history-input", type=Path, required=True)
    cold_pair.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/yambda_cold_pair_candidates.parquet"),
    )
    cold_pair.add_argument("--lookback-days", type=int, default=180)

    events = commands.add_parser(
        "prepare-events",
        help="Normalize Yambda events for automatic temporal features.",
    )
    events.add_argument("input", type=Path)
    events.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/yambda_events.parquet"),
    )

    unseen_pair = commands.add_parser(
        "unseen-pair-control",
        help="Keep queries whose positive items have no prior interaction.",
    )
    unseen_pair.add_argument("input", type=Path)
    unseen_pair.add_argument("--event-history-input", type=Path, required=True)
    unseen_pair.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/yambda_unseen_candidates.parquet"),
    )

    feature_search = commands.add_parser(
        "search-features",
        help="Generate and select temporal aggregate formulas against a manual control.",
    )
    feature_search.add_argument("input", type=Path)
    feature_search.add_argument("--event-history-input", type=Path, required=True)
    feature_search.add_argument(
        "--output-dir", type=Path, default=Path("outputs/feature-search")
    )
    feature_search.add_argument(
        "--entities", type=_parse_entities, default=("user", "item")
    )
    feature_search.add_argument(
        "--windows-days", type=_parse_windows_days, default=SEARCH_WINDOWS_DAYS
    )
    feature_search.add_argument("--feature-budget", type=int, default=12)
    feature_search.add_argument(
        "--split-quantiles",
        type=_parse_split_quantiles,
        default=(0.50, 0.70, 0.85),
    )
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

        if args.command == "cold-pair-control":
            report = prepare_cold_pair_candidates(
                args.input,
                args.history_input,
                args.output,
                args.lookback_days,
            )
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0

        if args.command == "prepare-events":
            report = prepare_yambda_event_history(args.input, args.output)
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0

        if args.command == "unseen-pair-control":
            report = prepare_unseen_pair_candidates(
                args.input, args.event_history_input, args.output
            )
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0

        if args.command == "search-features":
            report = run_feature_search(
                args.input,
                args.event_history_input,
                args.output_dir,
                args.entities,
                args.windows_days,
                args.feature_budget,
                args.split_quantiles,
            )
            print(
                json.dumps(
                    {
                        "manual_audit": report["manual"]["audit"]["metrics"],
                        "automatic_audit": report["automatic"]["audit"]["metrics"],
                        "popularity_audit": report["item_popularity_30d_audit"],
                        "selected_features": report["automatic"]["chosen_selection"]["features"],
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0

        if args.command == "search-windows":
            report = run_window_search(
                args.input,
                args.output_dir,
                args.history_input,
                args.base_windows,
                args.candidate_windows,
                args.user_column,
                args.item_column,
                args.timestamp_column,
                args.label_column,
                args.query_column,
                max_windows=args.max_windows,
            )
            print(json.dumps(report["best_by_budget"], indent=2, ensure_ascii=False))
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
            args.train_quantile,
            args.validation_quantile,
            {
                prefix: values
                for prefix in ("user", "item", "pair")
                if (values := getattr(args, f"{prefix}_windows_days")) is not None
            },
        )
        print(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))
        return 0
    except (DataContractError, OSError) as error:
        parser.error(str(error))
    return 2

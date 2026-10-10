from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from recsys_mvp.data import DataContractError, profile_data
from recsys_mvp.train import run_baseline


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
    _add_column_options(baseline, include_label=True)
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

        metrics = run_baseline(
            args.input,
            args.output_dir,
            args.user_column,
            args.item_column,
            args.timestamp_column,
            args.label_column,
            args.query_column,
        )
        print(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))
        return 0
    except (DataContractError, OSError) as error:
        parser.error(str(error))
    return 2

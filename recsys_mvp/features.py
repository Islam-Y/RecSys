from __future__ import annotations

from pathlib import Path

import duckdb

from recsys_mvp.data import DataContractError, quote_sql_string


WINDOWS_DAYS = (1, 7, 30)


def _count_expressions(prefix: str, windows_days: tuple[int, ...]) -> str:
    expressions = []
    for days in windows_days:
        expressions.append(
            f"""
            COUNT(h.event_time) FILTER (
                WHERE h.event_time >= e.event_time - INTERVAL '{days} days'
            ) AS {prefix}_events_{days}d
            """
        )
    return ",\n".join(expressions)


def _create_entity_features(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    key_columns: tuple[str, ...],
    prefix: str,
    windows_days: tuple[int, ...],
    include_trends: bool,
    include_recency: bool,
    include_popularity_baseline: bool = False,
) -> None:
    context_columns = ", ".join(f"e.{column}" for column in key_columns)
    history_condition = " AND ".join(
        f"h.{column} = e.{column}" for column in key_columns
    )
    group_columns = f"{context_columns}, e.event_time"
    join_columns = ", ".join((*key_columns, "event_time"))
    feature_column_names = [
        f"aggregated.{prefix}_events_{days}d"
        for days in windows_days
    ]
    if include_popularity_baseline:
        feature_column_names.append(f"aggregated.{prefix}_popularity_30d")
    feature_columns = ", ".join(feature_column_names)
    aggregate_expressions = [_count_expressions(prefix, windows_days)]
    if include_popularity_baseline:
        aggregate_expressions.append(
            f"""
            COUNT(h.event_time) FILTER (
                WHERE h.event_time >= e.event_time - INTERVAL '30 days'
            ) AS {prefix}_popularity_30d
            """
        )
    derived_columns = []
    if include_trends:
        aggregate_expressions.append(
            f"""
            COUNT(h.event_time) FILTER (
                WHERE h.event_time >= e.event_time - INTERVAL '14 days'
                  AND h.event_time < e.event_time - INTERVAL '7 days'
            ) AS {prefix}_events_prev_7d
            """
        )
        derived_columns.extend(
            [
                f"aggregated.{prefix}_events_prev_7d",
                f"""(
                    aggregated.{prefix}_events_7d
                    - aggregated.{prefix}_events_prev_7d
                ) / (
                    aggregated.{prefix}_events_7d
                    + aggregated.{prefix}_events_prev_7d + 1.0
                ) AS {prefix}_trend_7d""",
            ]
        )
    if include_recency:
        aggregate_expressions.append(
            f"""MAX(h.event_time) FILTER (
                WHERE h.event_time >= e.event_time
                    - INTERVAL '{max(windows_days)} days'
            ) AS {prefix}_last_event_time"""
        )
        derived_columns.append(
            f"""date_diff(
                'second', aggregated.{prefix}_last_event_time, e.event_time
            ) / 86400.0 AS {prefix}_days_since_last_listen_{max(windows_days)}d"""
        )
    selected_columns = ", ".join(
        value for value in (feature_columns, *derived_columns) if value
    )
    connection.execute(
        f"""
        CREATE TEMP TABLE {table_name} AS
        WITH contexts AS (
            SELECT DISTINCT {', '.join(key_columns)}, event_time
            FROM examples
        ),
        aggregated AS (
            SELECT
                {group_columns},
                {', '.join(aggregate_expressions)}
            FROM contexts AS e
            LEFT JOIN positive_history AS h
              ON {history_condition}
             AND h.event_time < e.event_time
             AND h.event_time >= e.event_time
                - INTERVAL '{max(max(windows_days), 30 if include_popularity_baseline else 1)} days'
            GROUP BY {group_columns}
        )
        SELECT e.row_id, {selected_columns}
        FROM examples AS e
        JOIN aggregated USING ({join_columns})
        """
    )


def build_feature_table(
    connection: duckdb.DuckDBPyConnection,
    history_input_path: Path | None = None,
    windows_days: tuple[int, ...] = WINDOWS_DAYS,
    include_trends: bool = False,
    include_recency: bool = False,
) -> None:
    windows_days = tuple(sorted(set(windows_days)))
    if not windows_days or any(days < 1 for days in windows_days):
        raise DataContractError("Feature windows must be positive whole days.")
    if include_trends and not {7, 14}.issubset(windows_days):
        raise DataContractError(
            "The 7-day trend requires both 7-day and 14-day windows."
        )
    if history_input_path is None:
        connection.execute(
            """
            CREATE TEMP VIEW positive_history AS
            SELECT user_id, item_id, event_time
            FROM examples
            WHERE label = 1
            """
        )
    else:
        history_path = history_input_path.expanduser().resolve()
        if not history_path.is_file():
            raise FileNotFoundError(f"History input file does not exist: {history_path}")
        connection.execute(
            f"""
            CREATE TEMP VIEW positive_history AS
            SELECT
                CAST(user_id AS VARCHAR) AS user_id,
                CAST(item_id AS VARCHAR) AS item_id,
                TRY_CAST(event_time AS TIMESTAMP) AS event_time
            FROM read_parquet({quote_sql_string(str(history_path))})
            """
        )
    _create_entity_features(
        connection,
        "user_features",
        ("user_id",),
        "user",
        windows_days,
        include_trends,
        include_recency,
    )
    _create_entity_features(
        connection,
        "item_features",
        ("item_id",),
        "item",
        windows_days,
        include_trends,
        include_recency,
        True,
    )
    _create_entity_features(
        connection,
        "pair_features",
        ("user_id", "item_id"),
        "pair",
        windows_days,
        include_trends,
        include_recency,
    )
    feature_projection_columns = []
    for prefix in ("user", "item", "pair"):
        entity_columns = [f"{prefix}_events_{days}d" for days in windows_days]
        if include_trends:
            entity_columns.extend(
                [f"{prefix}_events_prev_7d", f"{prefix}_trend_7d"]
            )
        if include_recency:
            entity_columns.append(
                f"{prefix}_days_since_last_listen_{max(windows_days)}d"
            )
        feature_projection_columns.extend(
            f"{prefix}_features.{column}" for column in entity_columns
        )
    feature_projection = ",\n            ".join(
        feature_projection_columns
    )
    connection.execute(
        f"""
        CREATE TEMP TABLE model_data AS
        SELECT
            e.row_id,
            e.user_id,
            e.item_id,
            e.query_id,
            e.event_time,
            e.label,
            item_features.item_popularity_30d,
            {feature_projection}
        FROM examples AS e
        JOIN user_features AS user_features USING (row_id)
        JOIN item_features AS item_features USING (row_id)
        JOIN pair_features AS pair_features USING (row_id)
        """
    )

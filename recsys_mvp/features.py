from __future__ import annotations

import duckdb
from pathlib import Path

from recsys_mvp.data import quote_sql_string


WINDOWS_DAYS = (1, 7, 30)
MAX_WINDOW_DAYS = max(WINDOWS_DAYS)


def _count_expressions(prefix: str) -> str:
    expressions = []
    for days in WINDOWS_DAYS:
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
) -> None:
    context_columns = ", ".join(f"e.{column}" for column in key_columns)
    history_condition = " AND ".join(
        f"h.{column} = e.{column}" for column in key_columns
    )
    group_columns = f"{context_columns}, e.event_time"
    join_columns = ", ".join((*key_columns, "event_time"))
    feature_columns = ", ".join(
        f"aggregated.{prefix}_events_{days}d"
        for days in WINDOWS_DAYS
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
                {_count_expressions(prefix)}
            FROM contexts AS e
            LEFT JOIN positive_history AS h
              ON {history_condition}
             AND h.event_time < e.event_time
             AND h.event_time >= e.event_time - INTERVAL '{MAX_WINDOW_DAYS} days'
            GROUP BY {group_columns}
        )
        SELECT e.row_id, {feature_columns}
        FROM examples AS e
        JOIN aggregated USING ({join_columns})
        """
    )


def build_feature_table(
    connection: duckdb.DuckDBPyConnection,
    history_input_path: Path | None = None,
) -> None:
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
    )
    _create_entity_features(
        connection,
        "item_features",
        ("item_id",),
        "item",
    )
    _create_entity_features(
        connection,
        "pair_features",
        ("user_id", "item_id"),
        "pair",
    )
    connection.execute(
        """
        CREATE TEMP TABLE model_data AS
        SELECT
            e.row_id,
            e.user_id,
            e.item_id,
            e.query_id,
            e.event_time,
            e.label,
            u.user_events_1d,
            u.user_events_7d,
            u.user_events_30d,
            i.item_events_1d,
            i.item_events_7d,
            i.item_events_30d,
            p.pair_events_1d,
            p.pair_events_7d,
            p.pair_events_30d
        FROM examples AS e
        JOIN user_features AS u USING (row_id)
        JOIN item_features AS i USING (row_id)
        JOIN pair_features AS p USING (row_id)
        """
    )

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb


class DataContractError(ValueError):
    """Raised when an input table does not match the MVP data contract."""


def quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def quote_sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def register_source(connection: duckdb.DuckDBPyConnection, input_path: Path) -> list[str]:
    path = input_path.expanduser().resolve()
    if not path.is_file():
        raise DataContractError(f"Input file does not exist: {path}")

    suffix = path.suffix.lower()
    path_sql = quote_sql_string(str(path))
    if suffix == ".parquet":
        reader = f"read_parquet({path_sql})"
    elif suffix in {".csv", ".tsv"}:
        delimiter = ","
        if suffix == ".tsv":
            delimiter = "\t"
        reader = f"read_csv_auto({path_sql}, header = true, delim = {quote_sql_string(delimiter)})"
    else:
        raise DataContractError("Supported input formats are CSV, TSV, and Parquet.")

    connection.execute(f"CREATE TEMP VIEW source_data AS SELECT * FROM {reader}")
    return [
        row[0]
        for row in connection.execute("DESCRIBE SELECT * FROM source_data").fetchall()
    ]


def profile_data(
    input_path: Path,
    user_column: str = "user_id",
    item_column: str = "item_id",
    timestamp_column: str = "timestamp",
    label_column: str = "label",
    query_column: str = "query_id",
) -> dict[str, Any]:
    connection = duckdb.connect()
    try:
        columns = register_source(connection, input_path)
        required = (user_column, item_column, timestamp_column)
        missing = [column for column in required if column not in columns]
        if missing:
            raise DataContractError(
                f"Missing required columns {missing}. Available columns: {columns}"
            )

        user = quote_identifier(user_column)
        item = quote_identifier(item_column)
        timestamp = quote_identifier(timestamp_column)
        summary = connection.execute(
            f"""
            SELECT
                COUNT(*) AS rows,
                COUNT(DISTINCT CAST({user} AS VARCHAR)) AS users,
                COUNT(DISTINCT CAST({item} AS VARCHAR)) AS items,
                MIN(TRY_CAST({timestamp} AS TIMESTAMP)) AS first_timestamp,
                MAX(TRY_CAST({timestamp} AS TIMESTAMP)) AS last_timestamp,
                COUNT(*) FILTER (WHERE {user} IS NULL) AS null_users,
                COUNT(*) FILTER (WHERE {item} IS NULL) AS null_items,
                COUNT(*) FILTER (WHERE TRY_CAST({timestamp} AS TIMESTAMP) IS NULL)
                    AS invalid_timestamps
            FROM source_data
            """
        ).fetchone()
        report: dict[str, Any] = {
            "input": str(input_path.expanduser().resolve()),
            "columns": columns,
            "rows": summary[0],
            "unique_users": summary[1],
            "unique_items": summary[2],
            "first_timestamp": summary[3],
            "last_timestamp": summary[4],
            "null_users": summary[5],
            "null_items": summary[6],
            "invalid_timestamps": summary[7],
        }

        if label_column in columns:
            label = quote_identifier(label_column)
            label_rows = connection.execute(
                f"""
                SELECT CAST({label} AS VARCHAR) AS label, COUNT(*) AS rows
                FROM source_data
                GROUP BY 1
                ORDER BY 1
                """
            ).fetchall()
            report["label_distribution"] = {
                str(value): count for value, count in label_rows
            }
        else:
            report["label_distribution"] = None

        if query_column in columns:
            query = quote_identifier(query_column)
            query_summary = connection.execute(
                f"""
                SELECT
                    COUNT(DISTINCT CAST({query} AS VARCHAR)),
                    AVG(candidate_count),
                    quantile_cont(candidate_count, 0.5),
                    MAX(candidate_count)
                FROM (
                    SELECT CAST({query} AS VARCHAR) AS query_id, COUNT(*) AS candidate_count
                    FROM source_data
                    WHERE {query} IS NOT NULL
                    GROUP BY 1
                )
                """
            ).fetchone()
            report["queries"] = {
                "count": query_summary[0] or 0,
                "average_candidates": query_summary[1],
                "median_candidates": query_summary[2],
                "max_candidates": query_summary[3],
            }
        else:
            report["queries"] = None

        return report
    finally:
        connection.close()


def create_examples_view(
    connection: duckdb.DuckDBPyConnection,
    columns: list[str],
    user_column: str,
    item_column: str,
    timestamp_column: str,
    label_column: str,
    query_column: str,
) -> None:
    requested = (user_column, item_column, timestamp_column, label_column)
    missing = [column for column in requested if column not in columns]
    if missing:
        raise DataContractError(
            f"Missing required columns {missing}. Available columns: {columns}"
        )

    user = quote_identifier(user_column)
    item = quote_identifier(item_column)
    timestamp = quote_identifier(timestamp_column)
    label = quote_identifier(label_column)
    query_expression = "CAST(NULL AS VARCHAR)"
    if query_column in columns:
        query_expression = f"CAST({quote_identifier(query_column)} AS VARCHAR)"
    elif query_column != "query_id":
        raise DataContractError(
            f"Query column {query_column!r} is not present. Available columns: {columns}"
        )

    connection.execute(
        f"""
        CREATE TEMP VIEW examples AS
        SELECT
            ROW_NUMBER() OVER () AS row_id,
            CAST({user} AS VARCHAR) AS user_id,
            CAST({item} AS VARCHAR) AS item_id,
            TRY_CAST({timestamp} AS TIMESTAMP) AS event_time,
            TRY_CAST({label} AS DOUBLE) AS label,
            {query_expression} AS query_id
        FROM source_data
        """
    )

    invalid_rows = connection.execute(
        """
        SELECT COUNT(*)
        FROM examples
        WHERE user_id IS NULL
           OR item_id IS NULL
           OR event_time IS NULL
           OR label IS NULL
           OR label NOT IN (0, 1)
        """
    ).fetchone()[0]
    if invalid_rows:
        raise DataContractError(
            f"{invalid_rows} rows have missing IDs/timestamps or labels outside 0/1. "
            "Map the source labels explicitly before training."
        )

    class_counts = dict(
        connection.execute(
            "SELECT CAST(label AS INTEGER), COUNT(*) FROM examples GROUP BY 1"
        ).fetchall()
    )
    if 0 not in class_counts or 1 not in class_counts:
        raise DataContractError(
            "Training requires both labels 0 and 1. Rows without an observed "
            "impression must not be assumed to be negative examples."
        )

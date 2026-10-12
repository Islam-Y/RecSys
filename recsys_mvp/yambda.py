from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb

from recsys_mvp.data import DataContractError, quote_sql_string


def prepare_yambda_event_history(
    input_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Normalize Yambda interactions for temporal feature generation."""
    source = input_path.expanduser().resolve()
    destination = output_path.expanduser().resolve()
    if not source.is_file():
        raise DataContractError(f"Input file does not exist: {source}")
    if source == destination:
        raise DataContractError("Event-history output must not overwrite the input.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    try:
        connection.execute("SET preserve_insertion_order = false")
        connection.execute(
            f"""
            COPY (
                SELECT
                    CAST(uid AS VARCHAR) AS user_id,
                    CAST(item_id AS VARCHAR) AS item_id,
                    make_timestamp(CAST(timestamp AS BIGINT) * 1000000) AS event_time,
                    CASE
                        WHEN event_type = 'listen' AND played_ratio_pct > 50
                            THEN 'listen_plus'
                        WHEN event_type = 'listen' AND played_ratio_pct <= 50
                            THEN 'listen_short'
                        ELSE event_type
                    END AS event_kind,
                    CAST(played_ratio_pct AS DOUBLE) AS played_ratio_pct
                FROM read_parquet({quote_sql_string(str(source))})
            ) TO {quote_sql_string(str(destination))}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        counts = dict(
            connection.execute(
                f"""
                SELECT event_kind, COUNT(*)
                FROM read_parquet({quote_sql_string(str(destination))})
                GROUP BY event_kind
                """
            ).fetchall()
        )
        report = {
            "source": str(source),
            "output": str(destination),
            "event_counts": counts,
            "time_rule": "source timestamps are seconds with arbitrary origin",
        }
        destination.with_suffix(destination.suffix + ".metadata.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return report
    finally:
        connection.close()


def prepare_unseen_pair_candidates(
    input_path: Path,
    event_history_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Keep queries whose positive items have no prior interaction of any kind."""
    source = input_path.expanduser().resolve()
    history = event_history_path.expanduser().resolve()
    destination = output_path.expanduser().resolve()
    for path in (source, history):
        if not path.is_file():
            raise DataContractError(f"Input file does not exist: {path}")
    if destination in {source, history}:
        raise DataContractError("Unseen-pair output must not overwrite an input.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    try:
        connection.execute(
            f"""
            CREATE TEMP TABLE seen_positive_queries AS
            SELECT DISTINCT c.query_id
            FROM read_parquet({quote_sql_string(str(source))}) AS c
            JOIN read_parquet({quote_sql_string(str(history))}) AS h
              ON c.user_id = h.user_id
             AND c.item_id = h.item_id
             AND h.event_time < c.timestamp
            WHERE c.label = 1
            """
        )
        excluded_queries = connection.execute(
            "SELECT COUNT(*) FROM seen_positive_queries"
        ).fetchone()[0]
        connection.execute(
            f"""
            COPY (
                SELECT c.*
                FROM read_parquet({quote_sql_string(str(source))}) AS c
                ANTI JOIN seen_positive_queries USING (query_id)
            ) TO {quote_sql_string(str(destination))}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        rows, queries, positives = connection.execute(
            f"""
            SELECT COUNT(*), COUNT(DISTINCT query_id), SUM(label)
            FROM read_parquet({quote_sql_string(str(destination))})
            """
        ).fetchone()
        report = {
            "source": str(source),
            "event_history": str(history),
            "output": str(destination),
            "excluded_queries": excluded_queries,
            "rows": rows,
            "queries": queries,
            "positives": positives,
            "rule": "exclude query if any positive item had any prior interaction",
        }
        destination.with_suffix(destination.suffix + ".metadata.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return report
    finally:
        connection.close()


def prepare_cold_pair_candidates(
    input_path: Path,
    history_input_path: Path,
    output_path: Path,
    lookback_days: int = 180,
) -> dict[str, Any]:
    """Keep queries whose positive items have no prior Listen+ in the lookback."""
    source = input_path.expanduser().resolve()
    history = history_input_path.expanduser().resolve()
    destination = output_path.expanduser().resolve()
    if lookback_days < 1:
        raise DataContractError("--lookback-days must be at least 1.")
    for path in (source, history):
        if not path.is_file():
            raise DataContractError(f"Input file does not exist: {path}")
    if destination in {source, history}:
        raise DataContractError("Cold-pair output must not overwrite an input.")
    destination.parent.mkdir(parents=True, exist_ok=True)

    connection = duckdb.connect()
    try:
        connection.execute(
            f"""
            CREATE TEMP TABLE repeated_queries AS
            SELECT DISTINCT c.query_id
            FROM read_parquet({quote_sql_string(str(source))}) AS c
            JOIN read_parquet({quote_sql_string(str(history))}) AS h
              ON c.user_id = h.user_id
             AND c.item_id = h.item_id
             AND h.event_time < c.timestamp
             AND h.event_time >= c.timestamp - INTERVAL '{lookback_days} days'
            WHERE c.label = 1
            """
        )
        excluded_queries = connection.execute(
            "SELECT COUNT(*) FROM repeated_queries"
        ).fetchone()[0]
        connection.execute(
            f"""
            COPY (
                SELECT c.*
                FROM read_parquet({quote_sql_string(str(source))}) AS c
                ANTI JOIN repeated_queries USING (query_id)
            ) TO {quote_sql_string(str(destination))}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        rows, queries, positives = connection.execute(
            f"""
            SELECT COUNT(*), COUNT(DISTINCT query_id), SUM(label)
            FROM read_parquet({quote_sql_string(str(destination))})
            """
        ).fetchone()
        report = {
            "source": str(source),
            "history_input": str(history),
            "output": str(destination),
            "lookback_days": lookback_days,
            "excluded_queries": excluded_queries,
            "rows": rows,
            "queries": queries,
            "positives": positives,
            "rule": (
                "exclude each query with any positive item previously "
                "listened to within lookback"
            ),
        }
        destination.with_suffix(destination.suffix + ".metadata.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return report
    finally:
        connection.close()


def prepare_yambda_candidates(
    input_path: Path,
    candidate_output: Path,
    history_output: Path,
    query_sample_modulo: int = 5_000,
    negatives_per_query: int = 20,
    seed: int = 42,
    negative_strategy: str = "mixed",
    popular_pool_size: int = 50_000,
    query_seed: int | None = None,
    popular_pool_cutoff_seconds: int | None = None,
    catalog_cutoff_seconds: int | None = None,
    unseen_positive_only: bool = False,
    excluded_query_inputs: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Prepare a small sampled next-listen ranking task from Yambda-50M.

    Positive query timestamps use query_seed; negatives use seed. Negatives are
    sampled from the available catalog and must be unseen by the user before
    each query. They are proxy negatives, not observed non-clicks.
    """
    source_path = input_path.expanduser().resolve()
    if not source_path.is_file():
        raise DataContractError(f"Input file does not exist: {source_path}")
    if query_sample_modulo < 1:
        raise DataContractError("--query-sample-modulo must be at least 1.")
    if negatives_per_query < 1:
        raise DataContractError("--negatives-per-query must be at least 1.")
    if negative_strategy not in {"uniform", "popular", "mixed"}:
        raise DataContractError(
            "--negative-strategy must be one of: uniform, popular, mixed."
        )
    if popular_pool_size < 1:
        raise DataContractError("--popular-pool-size must be at least 1.")
    query_seed = seed if query_seed is None else query_seed

    candidate_path = candidate_output.expanduser().resolve()
    history_path = history_output.expanduser().resolve()
    if source_path in {candidate_path, history_path}:
        raise DataContractError("Output paths must not overwrite the input dataset.")
    if candidate_path == history_path:
        raise DataContractError("Candidate and history outputs must be different files.")
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.unlink(missing_ok=True)
    history_path.unlink(missing_ok=True)

    connection = duckdb.connect()
    try:
        connection.execute("SET preserve_insertion_order = false")
        if unseen_positive_only:
            connection.execute("SET threads = 4")
            connection.execute("SET memory_limit = '6GB'")
        connection.execute(
            f"""
            CREATE TEMP VIEW yambda_source AS
            SELECT
                CAST(uid AS UBIGINT) AS user_id,
                CAST(item_id AS UBIGINT) AS item_id,
                CAST(timestamp AS UBIGINT) AS event_time_seconds,
                CAST(event_type AS VARCHAR) AS event_type,
                TRY_CAST(played_ratio_pct AS DOUBLE) AS played_ratio_pct
            FROM read_parquet({quote_sql_string(str(source_path))})
            """
        )
        source_summary = connection.execute(
            """
            SELECT
                COUNT(*),
                COUNT(DISTINCT user_id),
                COUNT(DISTINCT item_id),
                MIN(event_time_seconds),
                MAX(event_time_seconds),
                COUNT(*) FILTER (
                    WHERE event_type = 'listen' AND played_ratio_pct > 50
                )
            FROM yambda_source
            """
        ).fetchone()
        event_counts = dict(
            connection.execute(
                "SELECT event_type, COUNT(*) FROM yambda_source GROUP BY 1 ORDER BY 1"
            ).fetchall()
        )

        connection.execute(
            """
            CREATE TEMP TABLE positive_history AS
            SELECT DISTINCT user_id, item_id, event_time_seconds
            FROM yambda_source
            WHERE event_type = 'listen' AND played_ratio_pct > 50
            """
        )
        history_event_count = connection.execute(
            "SELECT COUNT(*) FROM positive_history"
        ).fetchone()[0]
        if history_event_count == 0:
            raise DataContractError("Yambda input contains no Listen+ events (>50% played).")

        connection.execute(
            f"""
            CREATE TEMP TABLE all_selected_queries AS
            SELECT
                user_id,
                event_time_seconds,
                CAST(user_id AS VARCHAR) || ':' || CAST(event_time_seconds AS VARCHAR) AS query_id
            FROM positive_history
            GROUP BY user_id, event_time_seconds
            HAVING hash(user_id, event_time_seconds, {int(query_seed)}) % {int(query_sample_modulo)} = 0
            """
        )
        query_count = connection.execute(
            "SELECT COUNT(*) FROM all_selected_queries"
        ).fetchone()[0]
        if query_count < 3:
            raise DataContractError(
                f"Only {query_count} query timestamps were sampled. Lower "
                "--query-sample-modulo and try again."
            )

        derived_popularity_cutoff = connection.execute(
            "SELECT quantile_disc(event_time_seconds, 0.20) FROM all_selected_queries"
        ).fetchone()[0]
        popularity_cutoff = (
            derived_popularity_cutoff
            if popular_pool_cutoff_seconds is None
            else popular_pool_cutoff_seconds
        )
        connection.execute(
            f"""
            CREATE TEMP TABLE selected_queries AS
            SELECT * FROM all_selected_queries
            WHERE event_time_seconds > {int(popularity_cutoff)}
            """
        )
        warmup_query_count = query_count - connection.execute(
            "SELECT COUNT(*) FROM selected_queries"
        ).fetchone()[0]
        excluded_paths = tuple(path.expanduser().resolve() for path in excluded_query_inputs)
        if excluded_paths:
            for path in excluded_paths:
                if not path.is_file() or path in {candidate_path, history_path}:
                    raise DataContractError(f"Invalid excluded-query input: {path}")
            paths_sql = ", ".join(quote_sql_string(str(path)) for path in excluded_paths)
            connection.execute(
                f"CREATE TEMP TABLE excluded_queries AS "
                f"SELECT DISTINCT query_id FROM read_parquet([{paths_sql}])"
            )
            connection.execute(
                "DELETE FROM selected_queries USING excluded_queries "
                "WHERE selected_queries.query_id = excluded_queries.query_id"
            )
        if connection.execute("SELECT COUNT(*) FROM selected_queries").fetchone()[0] < 3:
            raise DataContractError(
                "Fewer than three candidate timestamps remain after the popularity-pool "
                "warm-up. Lower --query-sample-modulo and retry."
            )
        derived_train_cutoff = connection.execute(
            "SELECT quantile_disc(event_time_seconds, 0.70) FROM selected_queries"
        ).fetchone()[0]
        train_cutoff = (
            derived_train_cutoff
            if catalog_cutoff_seconds is None
            else catalog_cutoff_seconds
        )
        if train_cutoff <= popularity_cutoff:
            raise DataContractError(
                "--catalog-cutoff-seconds must be later than the popularity-pool cutoff."
            )
        connection.execute(
            f"""
            CREATE TEMP TABLE item_catalog AS
            SELECT
                item_id,
                first_seen_seconds,
                CAST(ROW_NUMBER() OVER (ORDER BY item_id) - 1 AS UBIGINT) AS catalog_index
            FROM (
                SELECT item_id, MIN(event_time_seconds) AS first_seen_seconds
                FROM yambda_source
                WHERE event_time_seconds <= {int(train_cutoff)}
                GROUP BY item_id
            )
            """
        )
        catalog_size = connection.execute(
            "SELECT COUNT(*) FROM item_catalog"
        ).fetchone()[0]
        if catalog_size < 2:
            raise DataContractError("The training-period item catalog is too small to sample negatives.")

        popular_catalog_size = 0
        if negative_strategy in {"popular", "mixed"}:
            connection.execute(
                f"""
                CREATE TEMP TABLE popular_catalog AS
                SELECT
                    item_id,
                    first_seen_seconds,
                    CAST(ROW_NUMBER() OVER (
                        ORDER BY positive_events DESC, item_id
                    ) - 1 AS UBIGINT) AS popular_index
                FROM (
                    SELECT catalog.item_id, catalog.first_seen_seconds,
                           COUNT(*) AS positive_events
                    FROM positive_history AS history
                    JOIN item_catalog AS catalog USING (item_id)
                    WHERE history.event_time_seconds <= {int(popularity_cutoff)}
                    GROUP BY catalog.item_id, catalog.first_seen_seconds
                )
                QUALIFY popular_index < {int(popular_pool_size)}
                """
            )
            popular_catalog_size = connection.execute(
                "SELECT COUNT(*) FROM popular_catalog"
            ).fetchone()[0]
            if popular_catalog_size == 0:
                raise DataContractError(
                    "No items are available for the popular negative pool. "
                    "Increase the sampled warm-up period or use uniform negatives."
                )
        else:
            connection.execute(
                """
                CREATE TEMP TABLE popular_catalog AS
                SELECT item_id, first_seen_seconds,
                       CAST(catalog_index AS UBIGINT) AS popular_index
                FROM item_catalog
                WHERE false
                """
            )

        connection.execute(
            """
            CREATE TEMP TABLE sampled_targets AS
            SELECT DISTINCT q.user_id, q.event_time_seconds, q.query_id, h.item_id
            FROM selected_queries AS q
            JOIN positive_history AS h USING (user_id, event_time_seconds)
            JOIN item_catalog AS catalog USING (item_id)
            """
        )
        connection.execute(
            """
            CREATE TEMP TABLE first_pair_events AS
            SELECT user_id, item_id, MIN(event_time_seconds) AS first_event_seconds
            FROM yambda_source GROUP BY user_id, item_id
            """
        )
        if unseen_positive_only:
            connection.execute(
                """
                CREATE TEMP TABLE repeated_target_queries AS
                SELECT DISTINCT target.query_id FROM sampled_targets AS target
                JOIN first_pair_events AS history USING (user_id, item_id)
                WHERE history.first_event_seconds < target.event_time_seconds
                """
            )
            connection.execute(
                "DELETE FROM sampled_targets USING repeated_target_queries "
                "WHERE sampled_targets.query_id = repeated_target_queries.query_id"
            )
        connection.execute(
            """
            CREATE TEMP TABLE candidate_queries AS
            SELECT DISTINCT user_id, event_time_seconds, query_id
            FROM sampled_targets
            """
        )
        usable_query_count = connection.execute(
            "SELECT COUNT(*) FROM candidate_queries"
        ).fetchone()[0]
        if usable_query_count < 3:
            raise DataContractError(
                "Fewer than three sampled queries have positive items in the "
                "training-period catalog. Lower --query-sample-modulo and retry."
            )

        proposals_per_query = negatives_per_query * 8
        if negative_strategy == "uniform":
            strategy_expression = "'uniform'"
        elif negative_strategy == "popular":
            strategy_expression = "'popular'"
        else:
            strategy_expression = "CASE WHEN slots.slot % 2 = 0 THEN 'popular' ELSE 'uniform' END"
        connection.execute(
            f"""
            CREATE TEMP TABLE sampled_negatives AS
            WITH proposals AS (
                SELECT
                    q.user_id,
                    q.event_time_seconds,
                    q.query_id,
                    slots.slot,
                    {strategy_expression} AS sampling_source,
                    hash(q.user_id, q.event_time_seconds, slots.slot, {int(seed)}) AS proposal_hash
                FROM candidate_queries AS q
                CROSS JOIN range(0, {int(proposals_per_query)}) AS slots(slot)
            ), mapped AS (
                SELECT p.user_id, p.event_time_seconds, p.query_id, p.slot,
                       p.sampling_source, c.item_id
                FROM proposals AS p
                JOIN item_catalog AS c
                  ON p.sampling_source = 'uniform'
                 AND c.catalog_index = p.proposal_hash % {int(catalog_size)}
                WHERE c.first_seen_seconds <= p.event_time_seconds
                UNION ALL
                SELECT p.user_id, p.event_time_seconds, p.query_id, p.slot,
                       p.sampling_source, c.item_id
                FROM proposals AS p
                JOIN popular_catalog AS c
                  ON p.sampling_source = 'popular'
                 AND c.popular_index = p.proposal_hash % {max(popular_catalog_size, 1)}
                WHERE c.first_seen_seconds <= p.event_time_seconds
            ), eligible AS (
                SELECT m.*
                FROM mapped AS m
                LEFT JOIN first_pair_events AS seen
                  ON seen.user_id = m.user_id AND seen.item_id = m.item_id
                 AND seen.first_event_seconds < m.event_time_seconds
                LEFT JOIN sampled_targets AS target
                  ON target.query_id = m.query_id AND target.item_id = m.item_id
                WHERE seen.user_id IS NULL AND target.query_id IS NULL
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY m.query_id, m.item_id ORDER BY m.slot
                ) = 1
            )
            SELECT user_id, event_time_seconds, query_id, item_id, sampling_source
            FROM eligible
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY query_id ORDER BY slot, item_id
            ) <= {int(negatives_per_query)}
            """
        )

        negative_count = connection.execute(
            "SELECT COUNT(*) FROM sampled_negatives"
        ).fetchone()[0]
        if negative_count == 0:
            raise DataContractError("Negative sampling returned no eligible candidates.")

        connection.execute(
            f"""
            COPY (
                SELECT
                    CAST(user_id AS VARCHAR) AS user_id,
                    CAST(item_id AS VARCHAR) AS item_id,
                    make_timestamp(CAST(event_time_seconds AS BIGINT) * 1000000) AS timestamp,
                    CAST(1 AS INTEGER) AS label,
                    query_id
                FROM sampled_targets
                UNION ALL
                SELECT
                    CAST(user_id AS VARCHAR) AS user_id,
                    CAST(item_id AS VARCHAR) AS item_id,
                    make_timestamp(CAST(event_time_seconds AS BIGINT) * 1000000) AS timestamp,
                    CAST(0 AS INTEGER) AS label,
                    query_id
                FROM sampled_negatives
            ) TO {quote_sql_string(str(candidate_path))}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        connection.execute(
            f"""
            COPY (
                SELECT
                    CAST(user_id AS VARCHAR) AS user_id,
                    CAST(item_id AS VARCHAR) AS item_id,
                    make_timestamp(CAST(event_time_seconds AS BIGINT) * 1000000) AS event_time
                FROM positive_history
            ) TO {quote_sql_string(str(history_path))}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

        result: dict[str, Any] = {
            "source": str(source_path),
            "dataset": "Yambda-50M flat multi_event",
            "license": "Apache-2.0; dataset card limits publication to scientific and research purposes",
            "source_statistics": {
                "rows": source_summary[0],
                "users": source_summary[1],
                "items": source_summary[2],
                "minimum_event_time_seconds": source_summary[3],
                "maximum_event_time_seconds": source_summary[4],
                "listen_plus_rows": source_summary[5],
                "event_types": event_counts,
            },
            "task": {
                "positive": "listen events with played_ratio_pct > 50",
                "negative": (
                    f"{negative_strategy} catalog items with no prior user interaction"
                ),
                "negative_semantics": "sampled unobserved items, not observed non-clicks or dislikes",
                "timestamp_conversion": (
                    "source timestamps are seconds with 5-second granularity; "
                    "converted to microseconds for DuckDB timestamps, with arbitrary origin"
                ),
                "sampling": {
                    "query_hash_modulo": query_sample_modulo,
                    "seed": seed,
                    "negative_sampling_seed": seed,
                    "query_seed": query_seed,
                    "unseen_positive_only": unseen_positive_only,
                    "excluded_query_inputs": [str(path) for path in excluded_paths],
                    "popular_pool_cutoff_seconds_requested": popular_pool_cutoff_seconds,
                    "catalog_cutoff_seconds_requested": catalog_cutoff_seconds,
                    "requested_negatives_per_query": negatives_per_query,
                    "proposal_multiplier": 8,
                    "negative_strategy": negative_strategy,
                    "popular_pool_size_requested": popular_pool_size,
                    "popular_pool_size_actual": popular_catalog_size,
                    "popular_pool_cutoff_event_time_seconds": popularity_cutoff,
                    "catalog_cutoff_event_time_seconds": train_cutoff,
                    "popular_pool_warmup_share": 0.20,
                    "warmup_query_timestamps": warmup_query_count,
                    "query_timestamps_sampled": query_count,
                    "queries_with_warm_positive_items": usable_query_count,
                    "positive_candidates": connection.execute(
                        "SELECT COUNT(*) FROM sampled_targets"
                    ).fetchone()[0],
                    "negative_candidates": negative_count,
                    "negative_candidates_by_source": dict(
                        connection.execute(
                            "SELECT sampling_source, COUNT(*) FROM sampled_negatives GROUP BY 1"
                        ).fetchall()
                    ),
                },
                "catalog": {
                    "eligible_item_count": catalog_size,
                    "training_cutoff_event_time_seconds": train_cutoff,
                },
            },
            "outputs": {
                "candidates": str(candidate_path),
                "positive_history": str(history_path),
            },
        }
        metadata_path = candidate_path.with_suffix(candidate_path.suffix + ".metadata.json")
        metadata_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        return result
    finally:
        connection.close()

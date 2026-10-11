from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb

from recsys_mvp.data import DataContractError, quote_sql_string


EVENT_KINDS = ("listen_plus", "listen_short", "like", "dislike")
ENTITIES = ("user", "item", "pair")
SEARCH_WINDOWS_DAYS = (1, 7, 30, 90, 180)


@dataclass(frozen=True)
class FeatureSpec:
    entity: str
    event_kind: str
    aggregate: str
    window_days: int

    @property
    def name(self) -> str:
        return (
            f"{self.entity}_{self.event_kind}_{self.aggregate}_"
            f"{self.window_days}d"
        )

    def as_dict(self) -> dict[str, str | int]:
        return {"name": self.name, **asdict(self)}


def generate_feature_specs(
    entities: tuple[str, ...] = ENTITIES,
    windows_days: tuple[int, ...] = SEARCH_WINDOWS_DAYS,
) -> tuple[FeatureSpec, ...]:
    if (
        not entities
        or len(set(entities)) != len(entities)
        or set(entities) - set(ENTITIES)
    ):
        raise DataContractError("Entities must be unique user, item, or pair values.")
    if not windows_days or any(days < 1 for days in windows_days):
        raise DataContractError("Search windows must be positive whole days.")
    if len(set(windows_days)) != len(windows_days):
        raise DataContractError("Search windows must be unique.")

    specs = []
    for entity in entities:
        for event_kind in EVENT_KINDS:
            for window_days in sorted(windows_days):
                specs.append(FeatureSpec(entity, event_kind, "count", window_days))
                if entity in {"user", "item"} and event_kind == "listen_plus":
                    counterpart = "distinct_items" if entity == "user" else "distinct_users"
                    specs.append(
                        FeatureSpec(entity, event_kind, counterpart, window_days)
                    )
                if entity in {"user", "item"} and event_kind.startswith("listen_"):
                    specs.append(
                        FeatureSpec(entity, event_kind, "mean_played", window_days)
                    )
    return tuple(specs)


def _aggregate_expression(spec: FeatureSpec) -> str:
    condition = (
        f"h.event_kind = '{spec.event_kind}' "
        f"AND h.event_time >= e.event_time - INTERVAL '{spec.window_days} days'"
    )
    if spec.aggregate == "count":
        aggregate = "COUNT(h.event_time)"
    elif spec.aggregate == "distinct_items":
        aggregate = "COUNT(DISTINCT h.item_id)"
    elif spec.aggregate == "distinct_users":
        aggregate = "COUNT(DISTINCT h.user_id)"
    elif spec.aggregate == "mean_played":
        aggregate = "AVG(h.played_ratio_pct)"
    else:
        raise DataContractError(f"Unsupported aggregate: {spec.aggregate}")
    return f"{aggregate} FILTER (WHERE {condition}) AS {spec.name}"


def build_generated_feature_table(
    connection: duckdb.DuckDBPyConnection,
    event_history_path: Path,
    specs: tuple[FeatureSpec, ...],
) -> None:
    path = event_history_path.expanduser().resolve()
    if not path.is_file():
        raise DataContractError(f"Event history does not exist: {path}")
    if not specs:
        raise DataContractError("At least one feature specification is required.")
    required = {"user_id", "item_id", "event_time", "event_kind", "played_ratio_pct"}
    columns = {
        row[0]
        for row in connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet({quote_sql_string(str(path))})"
        ).fetchall()
    }
    if missing := required - columns:
        raise DataContractError(f"Event history is missing columns: {sorted(missing)}")

    connection.execute(
        f"""
        CREATE TEMP VIEW event_history AS
        SELECT user_id, item_id, event_time, event_kind, played_ratio_pct
        FROM read_parquet({quote_sql_string(str(path))})
        """
    )
    entity_keys = {
        "user": ("user_id",),
        "item": ("item_id",),
        "pair": ("user_id", "item_id"),
    }
    present_entities = tuple(
        entity for entity in ENTITIES if any(spec.entity == entity for spec in specs)
    )
    for entity in present_entities:
        entity_specs = tuple(spec for spec in specs if spec.entity == entity)
        keys = entity_keys[entity]
        key_projection = ", ".join(keys)
        group_projection = ", ".join(f"e.{key}" for key in keys)
        join_condition = " AND ".join(f"h.{key} = e.{key}" for key in keys)
        expressions = ",\n                ".join(
            _aggregate_expression(spec) for spec in entity_specs
        )
        connection.execute(
            f"""
            CREATE TEMP TABLE generated_{entity}_features AS
            WITH contexts AS (
                SELECT DISTINCT {key_projection}, event_time FROM examples
            ), aggregated AS (
                SELECT {group_projection}, e.event_time,
                    {expressions}
                FROM contexts AS e
                LEFT JOIN event_history AS h
                  ON {join_condition}
                 AND h.event_time < e.event_time
                 AND h.event_time >= e.event_time
                     - INTERVAL '{max(s.window_days for s in entity_specs)} days'
                GROUP BY {group_projection}, e.event_time
            )
            SELECT e.row_id, {', '.join('a.' + spec.name for spec in entity_specs)}
            FROM examples AS e
            JOIN aggregated AS a USING ({key_projection}, event_time)
            """
        )

    projections = ",\n            ".join(
        f"g_{spec.entity}.{spec.name}" for spec in specs
    )
    joins = "\n".join(
        f"JOIN generated_{entity}_features AS g_{entity} USING (row_id)"
        for entity in present_entities
    )
    connection.execute(
        f"""
        CREATE TEMP TABLE generated_model_data AS
        SELECT e.row_id, e.user_id, e.item_id, e.query_id, e.event_time,
            e.label, {projections}
        FROM examples AS e
        {joins}
        """
    )

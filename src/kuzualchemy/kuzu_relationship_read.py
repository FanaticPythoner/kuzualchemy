from __future__ import annotations
from functools import lru_cache
from typing import Any, Type
from atp_pipeline import DbStatement, normalize_relationship_db_direction
from .kuzu_session_rows import primary_key_fields


@lru_cache(maxsize=None)
def model_payload_fields(model_class: Type[Any]) -> frozenset[str]:
    fields = getattr(model_class, "model_fields", None)
    if not isinstance(fields, dict):
        raise TypeError(f"{model_class.__name__} has no field metadata")
    return frozenset(fields.keys())


def construct_model_from_db_payload(model_class: Type[Any], payload: dict[str, Any]) -> Any:
    construct = getattr(model_class, "model_construct", None)
    fields = getattr(model_class, "model_fields", None)
    if not isinstance(fields, dict):
        raise TypeError(f"{model_class.__name__} has no field metadata")
    filtered = {
        key: value
        for key, value in payload.items()
        if key in fields and not (value is None and _field_has_default_factory(fields[key]))
    }
    if construct is None:
        return model_class(**filtered)
    return construct(**filtered)


def _field_has_default_factory(field: Any) -> bool:
    return callable(getattr(field, "default_factory", None))


def relationship_read_name(relationship_class: Type[Any]) -> str:
    name = getattr(relationship_class, "__kuzu_rel_name__", None)
    if not isinstance(name, str) or not name:
        raise ValueError(f"{relationship_class.__name__} is not a registered Kuzu relationship")
    return name


def relationship_read_direction(relationship_class: Type[Any]) -> str:
    raw = getattr(relationship_class, "__kuzu_direction__", None)
    value = getattr(raw, "value", None) or getattr(raw, "name", None) or raw or "forward"
    try:
        return normalize_relationship_db_direction(str(value))
    except ValueError as exc:
        raise ValueError(
            f"{relationship_class.__name__} direction is unsupported: {value}"
        ) from exc


def relationship_read_pairs(relationship_class: Type[Any]) -> list[dict[str, str]]:
    pairs = getattr(relationship_class, "__kuzu_relationship_pairs__", None)
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"{relationship_class.__name__} relationship pairs are missing")
    rows: list[dict[str, str]] = []
    for pair in pairs:
        from_label = pair.get_from_name()
        to_label = pair.get_to_name()
        from_cls = getattr(pair, "from_node", None)
        to_cls = getattr(pair, "to_node", None)
        if not isinstance(from_label, str) or not from_label:
            raise ValueError(f"{relationship_class.__name__} relationship pair has no source label")
        if not isinstance(to_label, str) or not to_label:
            raise ValueError(f"{relationship_class.__name__} relationship pair has no target label")
        rows.append(
            {
                "from_label": from_label,
                "to_label": to_label,
                "from_key_field": primary_key_fields(_pair_node_class(from_cls, from_label))[0],
                "to_key_field": primary_key_fields(_pair_node_class(to_cls, to_label))[0],
            }
        )
    return rows

def relationship_endpoint_types(relationship_class: Type[Any]) -> dict[str, Type[Any]]:
    pairs = getattr(relationship_class, "__kuzu_relationship_pairs__", None)
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"{relationship_class.__name__} relationship pairs are missing")
    endpoint_types: dict[str, Type[Any]] = {}
    for pair in pairs:
        for attr, label in (
            ("from_node", pair.get_from_name()),
            ("to_node", pair.get_to_name()),
        ):
            if not isinstance(label, str) or not label:
                raise ValueError(f"{relationship_class.__name__} relationship pair has no label")
            model_class = _pair_node_class(getattr(pair, attr, None), label)
            existing = endpoint_types.get(label)
            if existing is not None and existing is not model_class:
                raise TypeError(f"relationship endpoint label maps to multiple classes: {label}")
            endpoint_types[label] = model_class
    return endpoint_types


def _pair_node_class(raw_node: Any, label: str) -> Type[Any]:
    if isinstance(raw_node, type):
        return raw_node
    from .kuzu_orm import get_node_by_name

    node_class = get_node_by_name(label)
    if node_class is None:
        raise TypeError(f"relationship endpoint label is not registered: {label}")
    return node_class


def can_use_native_relationship_read(state: Any) -> bool:
    return (
        hasattr(state.model_class, "__kuzu_rel_name__")
        and not state.order_by
        and state.limit_value is None
        and state.offset_value is None
        and not state.distinct
        and not state.select_fields
        and not state.aggregations
        and not state.group_by
        and state.having is None
        and not state.joins
        and not state.with_clauses
        and not state.return_raw
        and state.return_alias is None
        and state.return_model_class is None
        and not state.subqueries
        and not state.union_queries
    )


def relationship_read_filter_statements(state: Any) -> list[DbStatement]:
    filters = list(getattr(state, "filters", []) or [])
    if not filters:
        return []
    alias = str(getattr(state, "alias", "n") or "n")
    model_class = getattr(state, "model_class", None)
    model_alias = getattr(model_class, "__name__", alias)
    alias_map = {
        alias: alias,
        str(model_alias): alias,
        "from_node": "from_node",
        "to_node": "to_node",
    }
    statements: list[DbStatement] = []
    for idx, expression in enumerate(filters):
        to_cypher = getattr(expression, "to_cypher", None)
        get_parameters = getattr(expression, "get_parameters", None)
        if not callable(to_cypher) or not callable(get_parameters):
            raise TypeError("relationship filter lacks KuzuAlchemy expression methods")
        prefix = f"rel_filter_{idx}_"
        fragment = str(to_cypher(alias_map, prefix, relationship_alias=alias)).strip()
        if not fragment:
            raise ValueError("relationship filter fragment is empty")
        statements.append(DbStatement(fragment, _prefixed_parameters(get_parameters(), prefix)))
    return statements


def _prefixed_parameters(parameters: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in parameters.items()}


def materialize_endpoint_node(value: Any, endpoint_types: dict[str, Type[Any]]) -> Any:
    if not isinstance(value, dict):
        return value
    label = value.get("_label")
    if not isinstance(label, str) or not label:
        raise TypeError("relationship endpoint payload missing _label")
    from .kuzu_orm import KuzuNodeBase
    model_class = endpoint_types.get(label)
    if model_class is None:
        raise TypeError(f"relationship endpoint label is not registered: {label}")
    fields = getattr(model_class, "model_fields", None)
    if not isinstance(fields, dict):
        raise TypeError(f"relationship endpoint label has no field metadata: {label}")
    if issubclass(model_class, KuzuNodeBase):
        return construct_model_from_db_payload(model_class, value)
    pk_fields = primary_key_fields(model_class)
    if len(pk_fields) == 1:
        return value.get(pk_fields[0])
    return tuple(value.get(field) for field in pk_fields)

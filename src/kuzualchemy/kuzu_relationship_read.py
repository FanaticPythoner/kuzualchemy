from __future__ import annotations
from typing import Any, Type
from atp_pipeline import DbRelationshipPair, DbRelationshipSubset, DbWorkKind, DbWorkSpec
from .kuzu_session_rows import primary_key_fields


def relationship_read_work(
    *,
    relationship: str,
    alias: str,
    direction: str,
    pairs: list[dict[str, str]],
    pairs_subset: list[int],
) -> DbWorkSpec:
    pair_specs = [
        DbRelationshipPair(
            from_label=pair["from_label"],
            to_label=pair["to_label"],
            from_key_field=pair["from_key_field"],
            to_key_field=pair["to_key_field"],
        )
        for pair in pairs
    ]
    return DbWorkSpec(
        DbWorkKind.RELATIONSHIP_READ,
        relationship_subset=DbRelationshipSubset(
            relationship=relationship,
            alias=alias,
            direction=direction,
            pairs=pair_specs,
            pairs_subset=list(pairs_subset),
        ),
    )


def relationship_read_name(relationship_class: Type[Any]) -> str:
    name = getattr(relationship_class, "__kuzu_rel_name__", None)
    if not isinstance(name, str) or not name:
        raise ValueError(f"{relationship_class.__name__} is not a registered Kuzu relationship")
    return name


def relationship_read_direction(relationship_class: Type[Any]) -> str:
    raw = getattr(relationship_class, "__kuzu_direction__", None)
    value = getattr(raw, "value", None) or getattr(raw, "name", None) or raw or "forward"
    text = str(value).lower()
    if text in {"forward", "outgoing"}:
        return "forward"
    if text in {"backward", "incoming"}:
        return "backward"
    if text == "both":
        return "both"
    raise ValueError(f"{relationship_class.__name__} direction is unsupported: {value}")


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
                "from_key_field": primary_key_fields(from_cls)[0],
                "to_key_field": primary_key_fields(to_cls)[0],
            }
        )
    return rows


def can_use_native_relationship_read(state: Any) -> bool:
    return (
        hasattr(state.model_class, "__kuzu_rel_name__")
        and not state.filters
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


def materialize_endpoint_node(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    label = value.get("_label")
    if not isinstance(label, str) or not label:
        raise TypeError("relationship endpoint payload missing _label")
    from .kuzu_orm import KuzuNodeBase, get_node_by_name
    model_class = get_node_by_name(label)
    if model_class is None:
        raise TypeError(f"relationship endpoint label is not registered: {label}")
    fields = getattr(model_class, "model_fields", None)
    if not isinstance(fields, dict):
        raise TypeError(f"relationship endpoint label has no field metadata: {label}")
    if issubclass(model_class, KuzuNodeBase):
        return model_class(**{key: item for key, item in value.items() if key in fields})
    pk_fields = primary_key_fields(model_class)
    if len(pk_fields) == 1:
        return value.get(pk_fields[0])
    return tuple(value.get(field) for field in pk_fields)

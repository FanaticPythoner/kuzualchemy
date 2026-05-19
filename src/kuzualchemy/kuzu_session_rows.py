from __future__ import annotations
from typing import Any
from .kuzu_orm import get_node_by_name

RelationshipRoute = tuple[str, str, str, str, str]

def _node_label(model_class: type[Any]) -> str:
    label = getattr(model_class, "__kuzu_node_name__", None)
    if not isinstance(label, str) or not label:
        raise ValueError(f"{model_class.__name__} is not a registered Kuzu node")
    return label

def _primary_key_fields(model_class: type[Any]) -> list[str]:
    getter = getattr(model_class, "get_primary_key_fields", None)
    if not callable(getter):
        raise ValueError(f"{model_class.__name__} has no primary key metadata")
    fields = getter()
    if not isinstance(fields, list) or not fields:
        raise ValueError(f"{model_class.__name__} primary key metadata is empty")
    return fields

def _node_row(instance: Any) -> dict[str, Any]:
    return dict(instance.model_dump(mode="python"))

def _node_delete_row(instance: Any) -> dict[str, Any]:
    return {field: getattr(instance, field) for field in _primary_key_fields(type(instance))}

def _pk_row(fields: list[str], value: Any) -> dict[str, Any]:
    if len(fields) == 1:
        return {fields[0]: value[0] if isinstance(value, tuple) else value}
    if not isinstance(value, tuple) or len(value) != len(fields):
        raise ValueError("composite primary key value must match primary key field count")
    return {field: value[index] for index, field in enumerate(fields)}

def _relationship_row(instance: Any) -> dict[str, Any]:
    row = dict(instance.model_dump(mode="python", exclude={"from_node", "to_node"}))
    source = _endpoint(instance.from_node, type(instance), "from")
    target = _endpoint(instance.to_node, type(instance), "to")
    row.update({"from_label": source[0], "to_label": target[0], "from_pk_field": source[1], "to_pk_field": target[1], "from_pk": source[2], "to_pk": target[2]})
    return row

def _relationship_update_row(instance: Any, fields: list[str]) -> dict[str, Any]:
    source = _endpoint(instance.from_node, type(instance), "from")
    target = _endpoint(instance.to_node, type(instance), "to")
    row = {
        "from_label": source[0],
        "to_label": target[0],
        "from_pk_field": source[1],
        "to_pk_field": target[1],
        "from_pk": source[2],
        "to_pk": target[2],
    }
    row.update({field: getattr(instance, field) for field in fields})
    return row

def _relationship_route(rel_cls: type[Any], row: dict[str, Any]) -> RelationshipRoute:
    return (
        getattr(rel_cls, "__kuzu_rel_name__"),
        str(row["from_label"]),
        str(row["to_label"]),
        str(row["from_pk_field"]),
        str(row["to_pk_field"]),
    )

def relationship_route(rel_cls: type[Any], row: dict[str, Any]) -> RelationshipRoute:
    return _relationship_route(rel_cls, row)

def relationship_row(instance: Any) -> dict[str, Any]:
    return _relationship_row(instance)

def relationship_update_row(instance: Any, fields: list[str]) -> dict[str, Any]:
    return _relationship_update_row(instance, fields)

def primary_key_fields(model_class: type[Any]) -> list[str]:
    return _primary_key_fields(model_class)

def _endpoint(value: Any, rel_cls: type[Any], side: str) -> tuple[str, str, Any]:
    if hasattr(type(value), "__kuzu_node_name__"):
        cls = type(value)
        field = _primary_key_fields(cls)[0]
        return _node_label(cls), field, getattr(value, field)
    pairs = getattr(rel_cls, "__kuzu_relationship_pairs__", [])
    if len(pairs) != 1:
        raise ValueError(f"{rel_cls.__name__} raw {side} endpoint requires one relationship pair")
    label = pairs[0].get_from_name() if side == "from" else pairs[0].get_to_name()
    node_cls = get_node_by_name(label)
    if node_cls is None:
        raise ValueError(f"{rel_cls.__name__} endpoint label is not registered: {label}")
    return label, _primary_key_fields(node_cls)[0], value

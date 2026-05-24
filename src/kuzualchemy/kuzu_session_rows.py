from __future__ import annotations
from functools import cache
from typing import Any
from .kuzu_orm import get_node_by_name

RelationshipRoute = tuple[str, str, str, str, str]
_RELATIONSHIP_ENDPOINT_FIELDS = frozenset({"from_node", "to_node"})
RelationshipEndpointMetadata = tuple[str, str, str, str]
EndpointRow = tuple[str, str, Any]

@cache
def _node_label(model_class: type[Any]) -> str:
    label = getattr(model_class, "__kuzu_node_name__", None)
    if not isinstance(label, str) or not label:
        raise ValueError(f"{model_class.__name__} is not a registered Kuzu node")
    return label

@cache
def _primary_key_fields(model_class: type[Any]) -> list[str]:
    getter = getattr(model_class, "get_primary_key_fields", None)
    if not callable(getter):
        raise ValueError(f"{model_class.__name__} has no primary key metadata")
    fields = getter()
    if not isinstance(fields, list) or not fields:
        raise ValueError(f"{model_class.__name__} primary key metadata is empty")
    return fields

@cache
def _node_merge_policies(model_class: type[Any]) -> dict[str, str]:
    getter = getattr(model_class, "get_all_kuzu_metadata", None)
    if not callable(getter):
        return {}
    policies: dict[str, str] = {}
    for field, metadata in getter().items():
        policy = getattr(metadata, "atp_merge_policy", None)
        if policy is not None:
            policies[field] = str(policy)
    return policies

@cache
def _relationship_pair_metadata(rel_cls: type[Any]) -> tuple[RelationshipEndpointMetadata, ...]:
    pairs = getattr(rel_cls, "__kuzu_relationship_pairs__", [])
    if not pairs:
        raise ValueError(f"{rel_cls.__name__} endpoint metadata is empty")
    metadata: list[RelationshipEndpointMetadata] = []
    for pair in pairs:
        from_label = pair.get_from_name()
        to_label = pair.get_to_name()
        from_node_cls = get_node_by_name(from_label)
        to_node_cls = get_node_by_name(to_label)
        if from_node_cls is None:
            raise ValueError(f"{rel_cls.__name__} endpoint label is not registered: {from_label}")
        if to_node_cls is None:
            raise ValueError(f"{rel_cls.__name__} endpoint label is not registered: {to_label}")
        metadata.append((
            from_label,
            to_label,
            _primary_key_fields(from_node_cls)[0],
            _primary_key_fields(to_node_cls)[0],
        ))
    return tuple(metadata)

def _node_endpoint(value: Any) -> EndpointRow | None:
    if hasattr(type(value), "__kuzu_node_name__"):
        cls = type(value)
        field = _primary_key_fields(cls)[0]
        return _node_label(cls), field, getattr(value, field)
    return None

def _relationship_endpoint_metadata(
    rel_cls: type[Any],
    from_node: Any,
    to_node: Any,
) -> tuple[EndpointRow, EndpointRow]:
    from_endpoint = _node_endpoint(from_node)
    to_endpoint = _node_endpoint(to_node)
    pairs = _relationship_pair_metadata(rel_cls)

    if from_endpoint is not None and to_endpoint is not None:
        candidates = [
            pair for pair in pairs
            if pair[0] == from_endpoint[0] and pair[1] == to_endpoint[0]
        ]
    elif from_endpoint is not None:
        candidates = [pair for pair in pairs if pair[0] == from_endpoint[0]]
    elif to_endpoint is not None:
        candidates = [pair for pair in pairs if pair[1] == to_endpoint[0]]
    else:
        candidates = list(pairs)

    if len(candidates) != 1:
        raise ValueError(
            f"{rel_cls.__name__} endpoint route is ambiguous: "
            f"from={type(from_node).__name__}, to={type(to_node).__name__}, candidates={len(candidates)}"
        )

    from_label, to_label, from_key, to_key = candidates[0]
    return (
        from_endpoint or (from_label, from_key, from_node),
        to_endpoint or (to_label, to_key, to_node),
    )

def _model_row(instance: Any, exclude: frozenset[str] = frozenset()) -> dict[str, Any]:
    data = getattr(instance, "__dict__", None)
    if not isinstance(data, dict):
        return dict(instance.model_dump(mode="python", exclude=exclude))
    if not exclude:
        return dict(data)
    return {field: value for field, value in data.items() if field not in exclude}

def _node_row(instance: Any) -> dict[str, Any]:
    return _model_row(instance)

def _node_delete_row(instance: Any) -> dict[str, Any]:
    return {field: getattr(instance, field) for field in _primary_key_fields(type(instance))}

def _pk_row(fields: list[str], value: Any) -> dict[str, Any]:
    if len(fields) == 1:
        return {fields[0]: value[0] if isinstance(value, tuple) else value}
    if not isinstance(value, tuple) or len(value) != len(fields):
        raise ValueError("composite primary key value must match primary key field count")
    return {field: value[index] for index, field in enumerate(fields)}

def _relationship_row(instance: Any) -> dict[str, Any]:
    row = _model_row(instance, _RELATIONSHIP_ENDPOINT_FIELDS)
    from_endpoint, to_endpoint = _relationship_endpoint_metadata(
        type(instance),
        instance.from_node,
        instance.to_node,
    )
    row.update({
        "from_label": from_endpoint[0],
        "to_label": to_endpoint[0],
        "from_pk_field": from_endpoint[1],
        "to_pk_field": to_endpoint[1],
        "from_pk": from_endpoint[2],
        "to_pk": to_endpoint[2],
    })
    return row

def _relationship_update_row(instance: Any, fields: list[str]) -> dict[str, Any]:
    from_endpoint, to_endpoint = _relationship_endpoint_metadata(
        type(instance),
        instance.from_node,
        instance.to_node,
    )
    row = {
        "from_label": from_endpoint[0],
        "to_label": to_endpoint[0],
        "from_pk_field": from_endpoint[1],
        "to_pk_field": to_endpoint[1],
        "from_pk": from_endpoint[2],
        "to_pk": to_endpoint[2],
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

def node_merge_policies(model_class: type[Any]) -> dict[str, str]:
    return dict(_node_merge_policies(model_class))

def _endpoint(value: Any, rel_cls: type[Any], side: str) -> tuple[str, str, Any]:
    endpoint = _node_endpoint(value)
    if endpoint is not None:
        return endpoint
    pairs = _relationship_pair_metadata(rel_cls)
    if len(pairs) != 1:
        raise ValueError(f"{rel_cls.__name__} raw {side} endpoint requires one relationship pair")
    label = pairs[0][0] if side == "from" else pairs[0][1]
    key = pairs[0][2] if side == "from" else pairs[0][3]
    return label, key, value

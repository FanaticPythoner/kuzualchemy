from __future__ import annotations
from functools import lru_cache
from typing import Any
import uuid
from .constants import KuzuDefaultFunction
from .kuzu_function_types import DefaultFunctionBase
from .kuzu_orm import BulkInsertValueGeneratorRegistry

RelationshipRoute = tuple[str, str, str, str, str]
_RELATIONSHIP_ENDPOINT_FIELDS = frozenset({"from_node", "to_node"})
RelationshipEndpointMetadata = tuple[str, str, str, str]
RelationshipRouteSpec = dict[str, str]
ModelFieldSpec = dict[str, Any]
ModelUuidFieldSpec = tuple[str, bool, bool]
EndpointRow = tuple[str, str, Any]

def clear_session_row_metadata_caches() -> None:
    _node_label_cached.cache_clear()
    _primary_key_fields_cached.cache_clear()
    _node_merge_policies_cached.cache_clear()
    _relationship_pair_metadata_cached.cache_clear()
    _relationship_route_specs_cached.cache_clear()
    _relationship_route_for_labels_cached.cache_clear()
    _model_field_specs_cached.cache_clear()
    _model_uuid_field_specs_cached.cache_clear()

def _node_label(model_class: type[Any]) -> str:
    return _node_label_cached(model_class)

@lru_cache(maxsize=None)
def _node_label_cached(model_class: type[Any]) -> str:
    label = getattr(model_class, "__kuzu_node_name__", None)
    if not isinstance(label, str) or not label:
        raise ValueError(f"{model_class.__name__} is not a registered Kuzu node")
    return label

def _primary_key_fields(model_class: type[Any]) -> list[str]:
    return list(_primary_key_fields_cached(model_class))

@lru_cache(maxsize=None)
def _primary_key_fields_cached(model_class: type[Any]) -> tuple[str, ...]:
    getter = getattr(model_class, "get_primary_key_fields", None)
    if not callable(getter):
        raise ValueError(f"{model_class.__name__} has no primary key metadata")
    fields = getter()
    if not isinstance(fields, list) or not fields:
        raise ValueError(f"{model_class.__name__} primary key metadata is empty")
    if not all(isinstance(field, str) and field for field in fields):
        raise ValueError(f"{model_class.__name__} primary key metadata contains invalid fields")
    return tuple(fields)

def _node_merge_policies(model_class: type[Any]) -> dict[str, str]:
    return dict(_node_merge_policies_cached(model_class))

@lru_cache(maxsize=None)
def _node_merge_policies_cached(model_class: type[Any]) -> tuple[tuple[str, str], ...]:
    getter = getattr(model_class, "get_all_kuzu_metadata", None)
    if not callable(getter):
        return ()
    policies: dict[str, str] = {}
    for field, metadata in getter().items():
        policy = getattr(metadata, "atp_merge_policy", None)
        if policy is not None:
            policies[field] = str(policy)
    return tuple(sorted(policies.items()))

def _relationship_pair_metadata(rel_cls: type[Any]) -> tuple[RelationshipEndpointMetadata, ...]:
    return _relationship_pair_metadata_cached(rel_cls)

@lru_cache(maxsize=None)
def _relationship_pair_metadata_cached(rel_cls: type[Any]) -> tuple[RelationshipEndpointMetadata, ...]:
    pairs = getattr(rel_cls, "__kuzu_relationship_pairs__", [])
    if not pairs:
        raise ValueError(f"{rel_cls.__name__} endpoint metadata is empty")
    metadata: list[RelationshipEndpointMetadata] = []
    for pair in pairs:
        from_label = pair.get_from_name()
        to_label = pair.get_to_name()
        from_node_cls = _relationship_pair_node_class(pair, "from_node", from_label)
        to_node_cls = _relationship_pair_node_class(pair, "to_node", to_label)
        metadata.append((
            from_label,
            to_label,
            _primary_key_fields(from_node_cls)[0],
            _primary_key_fields(to_node_cls)[0],
        ))
    return tuple(metadata)

@lru_cache(maxsize=None)
def _relationship_route_specs_cached(rel_cls: type[Any]) -> tuple[RelationshipRouteSpec, ...]:
    return tuple(
        {
            "from_label": from_label,
            "to_label": to_label,
            "from_key_field": from_key,
            "to_key_field": to_key,
        }
        for from_label, to_label, from_key, to_key in _relationship_pair_metadata(rel_cls)
    )

def _relationship_pair_node_class(pair: Any, attr: str, label: str) -> type[Any]:
    raw_node = getattr(pair, attr, None)
    if isinstance(raw_node, type):
        return raw_node
    from .kuzu_orm import get_node_by_name

    node_class = get_node_by_name(label)
    if node_class is None:
        raise TypeError(f"relationship endpoint label is not registered: {label}")
    return node_class

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
    route = _relationship_route_for_labels_cached(
        rel_cls,
        None if from_endpoint is None else from_endpoint[0],
        None if to_endpoint is None else to_endpoint[0],
    )
    return (
        from_endpoint or (route[0], route[2], from_node),
        to_endpoint or (route[1], route[3], to_node),
    )

@lru_cache(maxsize=None)
def _relationship_route_for_labels_cached(
    rel_cls: type[Any],
    from_label: str | None,
    to_label: str | None,
) -> RelationshipEndpointMetadata:
    candidates = []
    for candidate in _relationship_pair_metadata(rel_cls):
        if from_label is not None and candidate[0] != from_label:
            continue
        if to_label is not None and candidate[1] != to_label:
            continue
        candidates.append(candidate)
    if len(candidates) != 1:
        raise ValueError(
            f"{rel_cls.__name__} endpoint route is ambiguous: "
            f"from={from_label or '*'} to={to_label or '*'} candidates={len(candidates)}"
        )
    return candidates[0]

def _model_row(instance: Any, exclude: frozenset[str] = frozenset()) -> dict[str, Any]:
    model_class = type(instance)
    data = getattr(instance, "__dict__", None)
    if not isinstance(data, dict):
        return _normalize_model_row_in_place(
            model_class,
            {
                field: _materialize_default_function(value)
                for field, value in instance.model_dump(mode="python", exclude=exclude).items()
            },
        )
    if not exclude:
        return _normalize_model_row_in_place(
            model_class,
            {
                field: _materialize_default_function(value)
                for field, value in data.items()
            },
        )
    return _normalize_model_row_in_place(
        model_class,
        {
            field: _materialize_default_function(value)
            for field, value in data.items()
            if field not in exclude
        },
    )

def normalize_model_row(model_class: type[Any], row: dict[str, Any]) -> dict[str, Any]:
    return _normalize_model_row_in_place(model_class, dict(row))

def _normalize_model_row_in_place(model_class: type[Any], row: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise TypeError("row must be a dict")
    for field, primary_key, not_null in _model_uuid_field_specs_cached(model_class):
        if field not in row:
            continue
        value = row[field]
        if isinstance(value, uuid.UUID) and value.int == 0:
            if primary_key or not_null:
                raise ValueError(f"UUID field {field} cannot use nil UUID for non-null storage")
            row[field] = None
    return row

def model_field_specs(model_class: type[Any]) -> list[dict[str, Any]]:
    return [dict(spec) for spec in _model_field_specs_cached(model_class)]

@lru_cache(maxsize=None)
def _model_field_specs_cached(model_class: type[Any]) -> tuple[ModelFieldSpec, ...]:
    getter = getattr(model_class, "get_all_kuzu_metadata", None)
    if not callable(getter):
        raise TypeError(f"{model_class.__name__} has no Kuzu field metadata")
    return tuple(
        {
            "field": field,
            "kuzu_type": str(getattr(metadata, "kuzu_type", "")),
            "primary_key": bool(getattr(metadata, "primary_key", False)),
            "not_null": bool(getattr(metadata, "not_null", False)),
            "auto_increment": bool(getattr(metadata, "auto_increment", False)),
        }
        for field, metadata in getter().items()
    )

@lru_cache(maxsize=None)
def _model_uuid_field_specs_cached(model_class: type[Any]) -> tuple[ModelUuidFieldSpec, ...]:
    return tuple(
        (
            str(spec["field"]),
            bool(spec["primary_key"]),
            bool(spec["not_null"]),
        )
        for spec in _model_field_specs_cached(model_class)
        if str(spec.get("kuzu_type", "")).upper() == "UUID"
    )

def _materialize_default_function(value: Any) -> Any:
    if isinstance(value, (KuzuDefaultFunction, DefaultFunctionBase)):
        return BulkInsertValueGeneratorRegistry.generate_value(value)
    return value

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

def auto_generation_fields(instance: Any) -> list[str]:
    getter = getattr(instance, "get_auto_increment_fields_needing_generation", None)
    if not callable(getter):
        return []
    fields = getter()
    if not isinstance(fields, list):
        raise TypeError("auto-increment metadata must be a list")
    return [field for field in fields if getattr(instance, field, None) is None]

def validate_explicit_null_primary_keys(
    instance: Any,
    manual_values: dict[str, Any],
) -> None:
    if not manual_values:
        return
    primary_keys = set(_primary_key_fields(type(instance)))
    null_keys = [
        field
        for field, value in manual_values.items()
        if field in primary_keys and value is None
    ]
    if null_keys:
        raise RuntimeError("violates non-null constraint of the primary key")

def node_create_spec(instance: Any) -> dict[str, Any]:
    generated = set(auto_generation_fields(instance))
    props = {
        field: value
        for field, value in _node_row(instance).items()
        if field not in generated
    }
    return {
        "label": _node_label(type(instance)),
        "props": props,
        "generated_fields": list(generated),
    }

def relationship_create_spec(instance: Any) -> dict[str, Any]:
    generated = set(auto_generation_fields(instance))
    row = _relationship_row(instance)
    route_fields = {
        "from_label",
        "to_label",
        "from_pk_field",
        "to_pk_field",
        "from_pk",
        "to_pk",
    }
    props = {
        field: value
        for field, value in row.items()
        if field not in generated and field not in route_fields
    }
    return {
        "rel_type": type(instance).__kuzu_rel_name__,
        "from_label": row["from_label"],
        "to_label": row["to_label"],
        "from_key_field": row["from_pk_field"],
        "to_key_field": row["to_pk_field"],
        "from_pk": row["from_pk"],
        "to_pk": row["to_pk"],
        "props": props,
        "generated_fields": list(generated),
    }

def apply_generated_values(
    instance: Any,
    payload: dict[str, Any],
    fields: list[str],
) -> None:
    for field in fields:
        if field not in payload:
            raise RuntimeError(f"generated payload missing field {field}")
        setattr(instance, field, payload[field])

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

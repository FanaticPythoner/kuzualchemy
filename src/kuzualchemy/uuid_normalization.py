from __future__ import annotations

import types
from typing import Any, Union, get_args, get_origin
import uuid
from weakref import WeakKeyDictionary

_NULL_UUID = uuid.UUID(int=0)
_NONE_TYPE = type(None)
_UNION_ORIGINS = (Union, types.UnionType) if hasattr(types, "UnionType") else (Union,)
_MODEL_UUID_FIELD_KIND_CACHE: "WeakKeyDictionary[type[Any], tuple[tuple[str, str], ...]]" = WeakKeyDictionary()


def clear_model_uuid_normalization_plan(model_class: type[Any]) -> None:
    _MODEL_UUID_FIELD_KIND_CACHE.pop(model_class, None)


def clear_all_uuid_normalization_plans() -> None:
    _MODEL_UUID_FIELD_KIND_CACHE.clear()


def uuid_field_kind(ann: object) -> str | None:
    if ann is uuid.UUID:
        return "uuid"
    origin = get_origin(ann)
    if origin is list:
        args = get_args(ann)
        if len(args) == 1 and args[0] is uuid.UUID:
            return "uuid_list"
        return None
    if origin in _UNION_ORIGINS:
        args = get_args(ann)
        if uuid.UUID in args and _NONE_TYPE in args:
            return "optional_uuid"
    return None


def coerce_uuid(v: Any) -> uuid.UUID:
    if isinstance(v, uuid.UUID):
        return v
    raise TypeError(f"Expected uuid.UUID, got {type(v)}")


def coerce_optional_uuid(v: Any) -> uuid.UUID | None:
    if v is None:
        return None
    return coerce_uuid(v)


def coerce_uuid_list(v: Any, *, field_name: str) -> list[uuid.UUID]:
    if v is None:
        raise TypeError(f"Field {field_name} expects list[uuid.UUID], got {type(v)}")
    if not isinstance(v, list):
        raise TypeError(f"Field {field_name} expects list[uuid.UUID], got {type(v)}")
    out: list[uuid.UUID] = []
    for elem in v:
        out.append(coerce_uuid(elem))
    return out


def _build_model_uuid_field_kinds(model_class: type[Any]) -> tuple[tuple[str, str], ...]:
    field_kinds: list[tuple[str, str]] = []
    for field_name, field_info in model_class.model_fields.items():
        kind = uuid_field_kind(field_info.annotation)
        if kind is not None:
            field_kinds.append((field_name, kind))
    return tuple(field_kinds)


def _get_model_uuid_field_kinds(model_class: type[Any]) -> tuple[tuple[str, str], ...]:
    cached = _MODEL_UUID_FIELD_KIND_CACHE.get(model_class)
    if cached is not None:
        return cached
    field_kinds = _build_model_uuid_field_kinds(model_class)
    _MODEL_UUID_FIELD_KIND_CACHE[model_class] = field_kinds
    return field_kinds


def normalize_uuid_fields_for_model(
    *,
    model_class: type[Any],
    data: dict[str, Any],
    null_uuid_sentinel: uuid.UUID | None = None,
) -> dict[str, Any]:
    if not hasattr(model_class, "model_fields"):
        raise TypeError("model_class must be a pydantic model class with model_fields")
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")

    for fname, kind in _get_model_uuid_field_kinds(model_class):
        if fname not in data:
            continue
        v = data[fname]
        if kind == "uuid":
            data[fname] = coerce_uuid(v)
            continue

        if kind == "optional_uuid":
            if null_uuid_sentinel is not None and v == null_uuid_sentinel:
                data[fname] = None
            else:
                data[fname] = coerce_optional_uuid(v)
            continue

        if kind == "uuid_list":
            data[fname] = coerce_uuid_list(v, field_name=fname)
            continue

    return data

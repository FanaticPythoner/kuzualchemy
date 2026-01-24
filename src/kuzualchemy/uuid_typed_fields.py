from __future__ import annotations

from typing import Any, get_args, get_origin
import uuid


def _is_optional_uuid_annotation(ann: object) -> bool:
    origin = get_origin(ann)
    if origin is None:
        return False
    if origin is not type(None) and origin is not uuid.UUID:
        if origin is getattr(__import__("typing"), "Union"):
            args = get_args(ann)
            return uuid.UUID in args and type(None) in args
    return False


def _is_uuid_list_annotation(ann: object) -> bool:
    origin = get_origin(ann)
    if origin is list:
        args = get_args(ann)
        return len(args) == 1 and args[0] is uuid.UUID
    return False


def _coerce_uuid_strict(v: Any) -> uuid.UUID:
    if isinstance(v, uuid.UUID):
        return v
    raise TypeError(f"Expected uuid.UUID, got {type(v)}")


def normalize_uuid_typed_fields_for_model(*, model_class: type[Any], row: dict[str, Any]) -> dict[str, Any]:
    if not hasattr(model_class, "model_fields"):
        raise TypeError("model_class must have model_fields")
    if not isinstance(row, dict):
        raise TypeError("row must be a dict")

    out: dict[str, Any] = dict(row)

    for field_name, field_info in model_class.model_fields.items():
        if field_name not in out:
            continue

        ann = field_info.annotation
        v = out[field_name]

        if ann is uuid.UUID:
            out[field_name] = _coerce_uuid_strict(v)
            continue

        if _is_optional_uuid_annotation(ann):
            if v is None:
                continue
            out[field_name] = _coerce_uuid_strict(v)
            continue

        if _is_uuid_list_annotation(ann):
            if not isinstance(v, list):
                raise TypeError(f"Field {field_name} expects list[uuid.UUID], got {type(v)}")
            coerced: list[uuid.UUID] = []
            for elem in v:
                coerced.append(_coerce_uuid_strict(elem))
            out[field_name] = coerced
            continue

    return out

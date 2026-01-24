from __future__ import annotations

from typing import Any, Union, get_args, get_origin
import uuid

_NULL_UUID = uuid.UUID(int=0)


def uuid_field_kind(ann: object) -> str | None:
    if ann is uuid.UUID:
        return "uuid"
    origin = get_origin(ann)
    if origin is list:
        args = get_args(ann)
        if len(args) == 1 and args[0] is uuid.UUID:
            return "uuid_list"
        return None
    if origin is Union:
        args = get_args(ann)
        if uuid.UUID in args and type(None) in args:
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

    out: dict[str, Any] = dict(data)
    for fname, fi in model_class.model_fields.items():
        if fname not in out:
            continue
        ann = fi.annotation
        kind = uuid_field_kind(ann)
        if kind is None:
            continue

        v = out[fname]
        if kind == "uuid":
            out[fname] = coerce_uuid(v)
            continue

        if kind == "optional_uuid":
            if null_uuid_sentinel is not None and v == null_uuid_sentinel:
                out[fname] = None
            else:
                out[fname] = coerce_optional_uuid(v)
            continue

        if kind == "uuid_list":
            out[fname] = coerce_uuid_list(v, field_name=fname)
            continue

    return out

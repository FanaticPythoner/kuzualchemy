from __future__ import annotations

import types
import uuid
from typing import Any, Union, get_args, get_origin

from atp_pipeline import normalize_kuzu_uuid_input_fields

_NONE_TYPE = type(None)
_UNION_ORIGINS = (Union, types.UnionType) if hasattr(types, "UnionType") else (Union,)


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


def normalize_uuid_fields_for_model(
    *,
    model_class: type[Any],
    data: dict[str, Any],
) -> dict[str, Any]:
    if not hasattr(model_class, "model_fields"):
        raise TypeError("model_class must be a pydantic model class with model_fields")
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")
    specs = [
        {"field": field_name, "kind": kind}
        for field_name, field_info in model_class.model_fields.items()
        for kind in [uuid_field_kind(field_info.annotation)]
        if kind is not None
    ]
    return normalize_kuzu_uuid_input_fields(data, specs)

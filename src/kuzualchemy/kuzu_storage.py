from __future__ import annotations

from enum import Enum
import struct
from typing import Any

from .constants import KuzuDataType
from .kuzu_orm import ArrayTypeSpecification


_FLOAT_STORAGE_FORMATS = {
    KuzuDataType.FLOAT.value: "!f",
    KuzuDataType.DOUBLE.value: "!d",
}


def _scalar_value(value: Any) -> Any:
    return value.value if isinstance(value, Enum) else value


def _storage_type_name(kuzu_type: KuzuDataType | str) -> str:
    value = kuzu_type.value if isinstance(kuzu_type, KuzuDataType) else kuzu_type
    if not isinstance(value, str) or not value:
        raise TypeError("Kuzu storage type must be a non-empty string or KuzuDataType")
    return value.upper()


def _float_storage_bytes(value: Any, *, storage_type: str, storage_format: str) -> bytes:
    try:
        return struct.pack(storage_format, float(_scalar_value(value)))
    except (OverflowError, TypeError, ValueError, struct.error) as exc:
        raise ValueError(f"{storage_type} value is outside its Kuzu storage domain: {value!r}") from exc


def kuzu_storage_equivalent(
    left: Any,
    right: Any,
    kuzu_type: KuzuDataType | str | ArrayTypeSpecification,
) -> bool:
    """Return whether two values map to the same declared Kuzu storage value.

    FLOAT and DOUBLE compare exact IEEE-754 storage bits. Arrays recurse over the
    declared element type. Other scalar types compare their enum-unwrapped values.
    """
    if left is None or right is None:
        return left is None and right is None
    if isinstance(kuzu_type, ArrayTypeSpecification):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            raise TypeError("Kuzu array storage values must be lists or tuples")
        return len(left) == len(right) and all(
            kuzu_storage_equivalent(left_item, right_item, kuzu_type.element_type)
            for left_item, right_item in zip(left, right, strict=True)
        )

    storage_type = _storage_type_name(kuzu_type)
    storage_format = _FLOAT_STORAGE_FORMATS.get(storage_type)
    if storage_format is not None:
        return _float_storage_bytes(
            left,
            storage_type=storage_type,
            storage_format=storage_format,
        ) == _float_storage_bytes(
            right,
            storage_type=storage_type,
            storage_format=storage_format,
        )
    return _scalar_value(left) == _scalar_value(right)


__all__ = ["kuzu_storage_equivalent"]

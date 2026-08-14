from __future__ import annotations

from enum import IntEnum
import struct

import pytest

from kuzualchemy import KuzuDataType, kuzu_storage_equivalent
from kuzualchemy.kuzu_orm import ArrayTypeSpecification


class _Code(IntEnum):
    FIRST = 1


def _float32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", value))[0]


def test_float_storage_equivalence_uses_exact_binary32_projection() -> None:
    assert kuzu_storage_equivalent(0.9, _float32(0.9), KuzuDataType.FLOAT)
    assert not kuzu_storage_equivalent(0.9, 0.91, KuzuDataType.FLOAT)


def test_float_storage_equivalence_preserves_signed_zero_bits() -> None:
    assert not kuzu_storage_equivalent(-0.0, 0.0, KuzuDataType.FLOAT)
    assert not kuzu_storage_equivalent(-0.0, 0.0, KuzuDataType.DOUBLE)


def test_float_storage_equivalence_preserves_nan_payload_bits() -> None:
    first = struct.unpack("!d", bytes.fromhex("7ff8000000000001"))[0]
    same = struct.unpack("!d", bytes.fromhex("7ff8000000000001"))[0]
    second = struct.unpack("!d", bytes.fromhex("7ff8000000000002"))[0]

    assert kuzu_storage_equivalent(first, same, KuzuDataType.DOUBLE)
    assert not kuzu_storage_equivalent(first, second, KuzuDataType.DOUBLE)


def test_array_storage_equivalence_recurses_without_length_caps() -> None:
    float_array = ArrayTypeSpecification(element_type=KuzuDataType.FLOAT)

    assert kuzu_storage_equivalent(
        [0.9, 0.25, None],
        [_float32(0.9), _float32(0.25), None],
        float_array,
    )
    assert not kuzu_storage_equivalent([0.9], [_float32(0.9), 0.25], float_array)


def test_non_float_storage_equivalence_unwraps_enums() -> None:
    assert kuzu_storage_equivalent(_Code.FIRST, 1, KuzuDataType.INT8)


def test_invalid_float_and_array_values_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="outside its Kuzu storage domain"):
        kuzu_storage_equivalent(1e100, 1e100, KuzuDataType.FLOAT)
    with pytest.raises(TypeError, match="must be lists or tuples"):
        kuzu_storage_equivalent("0.9", "0.9", ArrayTypeSpecification(KuzuDataType.FLOAT))

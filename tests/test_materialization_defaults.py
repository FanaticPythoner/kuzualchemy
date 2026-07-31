# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import IntEnum

import pytest

from kuzualchemy import KuzuBaseModel, KuzuDataType, kuzu_field, kuzu_int8enum, kuzu_node
from kuzualchemy.kuzu_relationship_read import construct_model_from_db_payload


@kuzu_node("DefaultFactoryNode")
class DefaultFactoryNode(KuzuBaseModel):
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    values: list[str] = kuzu_field(
        kuzu_type=KuzuDataType.ARRAY,
        element_type=KuzuDataType.STRING,
        default_factory=list,
    )


@kuzu_int8enum
class MaterializedState(IntEnum):
    READY = 1


@kuzu_node("EnumMaterializationNode")
class EnumMaterializationNode(KuzuBaseModel):
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    state: MaterializedState = kuzu_field(kuzu_type=KuzuDataType.INT8)


def test_construct_model_from_db_payload_preserves_default_factory_for_null_values() -> None:
    node = construct_model_from_db_payload(DefaultFactoryNode, {"id": 1, "values": None})

    assert node.values == []


def test_construct_model_from_db_payload_rehydrates_enum_without_model_validation() -> None:
    node = construct_model_from_db_payload(EnumMaterializationNode, {"id": 1, "state": 1})

    assert node.state is MaterializedState.READY


def test_construct_model_from_db_payload_rejects_unknown_enum_value() -> None:
    with pytest.raises(ValueError, match="Invalid value for field state"):
        construct_model_from_db_payload(EnumMaterializationNode, {"id": 1, "state": 2})

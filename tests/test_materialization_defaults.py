# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from kuzualchemy import KuzuBaseModel, KuzuDataType, kuzu_field, kuzu_node
from kuzualchemy.kuzu_relationship_read import construct_model_from_db_payload


@kuzu_node("DefaultFactoryNode")
class DefaultFactoryNode(KuzuBaseModel):
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    values: list[str] = kuzu_field(
        kuzu_type=KuzuDataType.ARRAY,
        element_type=KuzuDataType.STRING,
        default_factory=list,
    )


def test_construct_model_from_db_payload_preserves_default_factory_for_null_values() -> None:
    node = construct_model_from_db_payload(DefaultFactoryNode, {"id": 1, "values": None})

    assert node.values == []

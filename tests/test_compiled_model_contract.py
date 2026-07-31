from __future__ import annotations

from kuzualchemy.kuzu_orm import KuzuBaseModel


def test_pydantic_ignores_runtime_method_representation() -> None:
    ignored_types = KuzuBaseModel.model_config["ignored_types"]

    assert isinstance(
        KuzuBaseModel.get_auto_increment_fields_needing_generation,
        ignored_types,
    )

from __future__ import annotations

from kuzualchemy.kuzu_orm import KuzuBaseModel


def test_pydantic_ignores_runtime_method_representation() -> None:
    ignored_types = KuzuBaseModel.model_config["ignored_types"]

    assert isinstance(
        KuzuBaseModel.get_auto_increment_fields_needing_generation,
        ignored_types,
    )


def test_pydantic_ignores_foreign_cython_runtime_types() -> None:
    foreign_callable_type = type(
        "cython_function_or_method",
        (),
        {
            "__module__": "_cython_foreign",
            "__call__": lambda self, value: value,
        },
    )
    foreign_callable = foreign_callable_type()

    class ForeignRuntimeModel(KuzuBaseModel):
        value: int
        transform = foreign_callable

    assert foreign_callable_type in ForeignRuntimeModel.model_config["ignored_types"]
    assert ForeignRuntimeModel(value=7).transform(11) == 11

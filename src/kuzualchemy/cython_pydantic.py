# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""Define Pydantic model primitives compatible with foreign Cython runtimes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

if TYPE_CHECKING:
    from pydantic._internal._model_construction import (
        ModelMetaclass as _PydanticModelMetaclass,
    )
else:
    _PydanticModelMetaclass = type(BaseModel)


def _cython_runtime_types(namespace: dict[str, Any]) -> tuple[type[Any], ...]:
    """Return distinct Cython runtime types present in a class namespace.

    Complexity
    ----------
    O(n) time and O(k) space for n namespace values and k distinct runtime types.
    """
    runtime_types = (
        type(value)
        for value in namespace.values()
        if type(value).__module__.startswith("_cython_")
    )
    return tuple(dict.fromkeys(runtime_types))


class CythonModelMetaclass(_PydanticModelMetaclass):
    """Register each compiled namespace's exact Cython runtime types with Pydantic."""

    def __new__(
        mcls,
        cls_name: str,
        bases: tuple[type[Any], ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> type[Any]:
        model_namespace = dict(namespace)
        runtime_types = _cython_runtime_types(model_namespace)
        if runtime_types:
            local_config = dict(model_namespace.get("model_config", {}))
            inherited_types = (
                ignored_type
                for base in bases
                for ignored_type in getattr(base, "model_config", {}).get("ignored_types", ())
            )
            configured_types = local_config.get("ignored_types", ())
            local_config["ignored_types"] = tuple(
                dict.fromkeys((*inherited_types, *configured_types, *runtime_types))
            )
            model_namespace["model_config"] = local_config
        model_class = super().__new__(mcls, cls_name, bases, model_namespace, **kwargs)
        return cast(type[Any], model_class)


__all__ = ["CythonModelMetaclass"]

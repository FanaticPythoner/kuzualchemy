# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, Type
from enum import Enum
from pydantic import model_validator

from .enum_normalization import create_enum_converter, get_enum_lookups
from .kuzu_orm import KuzuBaseModel
from .uuid_normalization import normalize_uuid_fields_for_model


class BaseModel(KuzuBaseModel):
    """
    Base model with automatic enum conversion for Kuzu ORM.
    
    Provides automatic conversion of string values to enum instances
    for enum-typed fields during model validation.
    """
    
    @staticmethod
    def _get_enum_lookups(enum_type: Type[Enum]) -> tuple[Dict[str, Enum], Dict[Any, Enum]]:
        """
        Get or create cached lookup dictionaries for an enum type.

        Creates O(1) lookup maps for enum member names and values.

        Args:
            enum_type: The enum class to create lookups for

        Returns:
            Tuple of (names_dict, values_dict) for O(1) lookup
        """
        return get_enum_lookups(enum_type)

    @staticmethod
    def _create_enum_converter(enum_type: Type[Enum]) -> callable:
        """
        Create a high-performance converter function for enum elements.

        Returns a closure that captures cached lookups for O(1) performance.
        """
        return create_enum_converter(enum_type)

    @model_validator(mode='before')
    @classmethod
    def normalize_uuid_fields(cls: Type['BaseModel'], values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        return normalize_uuid_fields_for_model(model_class=cls, data=values)

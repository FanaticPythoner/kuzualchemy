# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Union, get_origin, get_args, Type, Dict
from enum import Enum
from pydantic import model_validator

from .kuzu_orm import KuzuBaseModel

# Module-level cache for enum lookups: {EnumClass: (names_dict, values_dict)}
_ENUM_CACHE: Dict[Type[Enum], tuple[Dict[str, Enum], Dict[Any, Enum]]] = {}

# Sentinel object to distinguish missing fields from None values
_MISSING = object()


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
        global _ENUM_CACHE
        if enum_type not in _ENUM_CACHE:
            names = {}
            values = {}
            for member in enum_type:
                names[member.name] = member
                values[member.value] = member
            _ENUM_CACHE[enum_type] = (names, values)
        return _ENUM_CACHE[enum_type]
    
    @model_validator(mode='before')
    @classmethod
    def convert_str_to_enum(cls: Type['BaseModel'], values: Any) -> Any:
        """
        Convert string values to enum instances for enum-typed fields.

        Single-pass O(n) algorithm with O(1) enum lookups using cached mappings.
        No exception handling, no fallbacks, no call stack inspection.

        Approach:
        1. Get type annotations (resolved or raw strings)
        2. For each field: extract enum type, perform cached lookup, convert value
        3. Raise ValueError immediately on invalid conversion

        Args:
            values: Input values dictionary

        Returns:
            Modified values with enum conversions applied

        Raises:
            ValueError: If a string value cannot be converted to the target enum
        """
        if not isinstance(values, dict):
            return values

        # Single loop through Pydantic's resolved field information
        for field_name, field_info in cls.model_fields.items():
            # Optimization #1: Single dictionary lookup with sentinel for missing fields
            value = values.get(field_name, _MISSING)
            if value is _MISSING or isinstance(value, Enum):
                continue

            # Get resolved type from Pydantic's field info
            field_type = field_info.annotation

            # Optimization #2 & #3: Conditional assignment and type checking
            if get_origin(field_type) is Union:
                enum_type = None
                union_args = get_args(field_type)
                has_none_type = type(None) in union_args

                for arg in union_args:
                    if isinstance(arg, type) and issubclass(arg, Enum):
                        enum_type = arg
                        break
                if enum_type is None:
                    continue

                # For Optional[Enum], skip None values (they should remain None)
                if value is None and has_none_type:
                    continue
                # No need to re-verify enum_type - already checked in loop
            else:
                enum_type = field_type
                if not (isinstance(enum_type, type) and issubclass(enum_type, Enum)):
                    continue

            # Optimization #4: Tuple unpacking for faster cache access
            names, value_map = BaseModel._get_enum_lookups(enum_type)

            # Direct value lookup (works for all types)
            member = value_map.get(value, _MISSING)
            if member is not _MISSING:
                values[field_name] = member
                continue

            # String-specific conversions
            if isinstance(value, str):
                # Name lookup
                member = names.get(value, _MISSING)
                if member is not _MISSING:
                    values[field_name] = member
                    continue

                # Optimization #5: Optimized numeric string conversion
                if len(value) > 0:
                    first_char = value[0]
                    if first_char.isdigit() or (len(value) > 1 and first_char == '-' and value[1].isdigit()):
                        numeric = int(value) if '.' not in value and 'e' not in value.lower() else float(value)
                        member = value_map.get(numeric, _MISSING)
                        if member is not _MISSING:
                            values[field_name] = member
                            continue

            # Invalid value - immediate error
            valid_names = list(names.keys())
            valid_values = [v for v in value_map.keys() if v is not None]
            raise ValueError(
                f"Invalid value for field {field_name}: {value} "
                f"Valid names: {valid_names}, valid values: {valid_values}"
            )

        return values

# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Union, get_origin, get_args, Type, Dict
from enum import Enum
from pydantic import model_validator
import uuid

from .kuzu_orm import KuzuBaseModel
from .uuid_normalization import normalize_uuid_fields_for_model

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

    @staticmethod
    def _create_enum_converter(enum_type: Type[Enum]) -> callable:
        """
        Create a high-performance converter function for enum elements.

        Returns a closure that captures cached lookups for O(1) performance.
        """
        names, value_map = BaseModel._get_enum_lookups(enum_type)

        def convert_element(elem: Any) -> Any:
            # Fast path: already correct enum instance
            if isinstance(elem, Enum) and elem.__class__ is enum_type:
                return elem

            # Direct value lookup
            member = value_map.get(elem, _MISSING)
            if member is not _MISSING:
                return member

            # String conversions
            if isinstance(elem, str):
                member = names.get(elem, _MISSING)
                if member is not _MISSING:
                    return member

                # Numeric string conversion
                if len(elem) > 0:
                    first_char = elem[0]
                    if first_char.isdigit() or (len(elem) > 1 and first_char == '-' and elem[1].isdigit()):
                        try:
                            numeric = int(elem) if '.' not in elem and 'e' not in elem.lower() else float(elem)
                            member = value_map.get(numeric, _MISSING)
                            if member is not _MISSING:
                                return member
                        except (ValueError, OverflowError):
                            pass

            raise ValueError(f"Invalid enum value: {elem}")

        return convert_element

    @model_validator(mode='before')
    @classmethod
    def normalize_uuid_fields(cls: Type['BaseModel'], values: Any) -> Any:
        if not isinstance(values, dict):
            return values

        # Only normalize keys present in values (avoid populating defaults here).
        present = {k: v for (k, v) in values.items() if k in cls.model_fields}
        norm = normalize_uuid_fields_for_model(model_class=cls, data=present)
        values.update(norm)
        return values

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

            # Fast path: Direct enum type (most common case)
            if isinstance(field_type, type) and issubclass(field_type, Enum):
                enum_type = field_type
                # Inline scalar conversion for maximum performance
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

                    # Optimized numeric string conversion
                    if len(value) > 0:
                        first_char = value[0]
                        if first_char.isdigit() or (len(value) > 1 and first_char == '-' and value[1].isdigit()):
                            try:
                                numeric = int(value) if '.' not in value and 'e' not in value.lower() else float(value)
                                member = value_map.get(numeric, _MISSING)
                                if member is not _MISSING:
                                    values[field_name] = member
                                    continue
                            except (ValueError, OverflowError):
                                pass

                # Invalid value - immediate error
                valid_names = list(names.keys())
                valid_values = list(filter(lambda v: v is not None, value_map.keys()))
                raise ValueError(
                    f"Invalid value for field {field_name}: {value} "
                    f"Valid names: {valid_names}, valid values: {valid_values}"
                )

            # Complex type analysis (Union, List, Tuple)
            enum_type = None
            is_sequence = False
            allow_none_elements = False

            origin = get_origin(field_type)
            if origin is Union:
                # Handle Optional[Enum] and Union types
                union_args = get_args(field_type)
                has_none_type = type(None) in union_args

                if len(union_args) >= 1:
                    arg0 = union_args[0]
                    if isinstance(arg0, type) and issubclass(arg0, Enum):
                        enum_type = arg0
                    elif len(union_args) >= 2:
                        arg1 = union_args[1]
                        if isinstance(arg1, type) and issubclass(arg1, Enum):
                            enum_type = arg1
                        elif len(union_args) >= 3:
                            arg2 = union_args[2]
                            if isinstance(arg2, type) and issubclass(arg2, Enum):
                                enum_type = arg2

                if enum_type is None:
                    continue

                # For Optional[Enum], skip None values (they should remain None)
                if value is None and has_none_type:
                    continue
            elif origin in (list, tuple):
                # Handle List[Enum], Tuple[Enum], List[Optional[Enum]]
                args = get_args(field_type)
                if not args:
                    continue

                inner_type = args[0]
                inner_origin = get_origin(inner_type)

                if inner_origin is Union:
                    # List[Optional[Enum]] case
                    inner_args = get_args(inner_type)
                    has_none = type(None) in inner_args

                    # Check first 3 args
                    if len(inner_args) >= 1:
                        arg0 = inner_args[0]
                        if isinstance(arg0, type) and issubclass(arg0, Enum):
                            enum_type = arg0
                            is_sequence = True
                            allow_none_elements = has_none
                        elif len(inner_args) >= 2:
                            arg1 = inner_args[1]
                            if isinstance(arg1, type) and issubclass(arg1, Enum):
                                enum_type = arg1
                                is_sequence = True
                                allow_none_elements = has_none
                elif isinstance(inner_type, type) and issubclass(inner_type, Enum):
                    # List[Enum] case
                    enum_type = inner_type
                    is_sequence = True

                if enum_type is None:
                    continue
            else:
                # Direct enum type
                enum_type = field_type
                if not (isinstance(enum_type, type) and issubclass(enum_type, Enum)):
                    continue

            # Conversion logic: sequence vs scalar
            if is_sequence:
                # Sequence conversion path
                if not isinstance(value, (list, tuple)):
                    continue  # Let Pydantic handle type coercion

                # FOR TOKEN #3: Single comprehension for sequence conversion
                converted = []
                for elem in value:
                    if allow_none_elements and elem is None:
                        converted.append(None)
                    elif isinstance(elem, Enum) and elem.__class__ is enum_type:
                        converted.append(elem)
                    else:
                        # Inline enum conversion for sequence elements
                        names, value_map = BaseModel._get_enum_lookups(enum_type)
                        member = value_map.get(elem, _MISSING)
                        if member is not _MISSING:
                            converted.append(member)
                        elif isinstance(elem, str):
                            member = names.get(elem, _MISSING)
                            if member is not _MISSING:
                                converted.append(member)
                            elif len(elem) > 0:
                                first_char = elem[0]
                                if first_char.isdigit() or (len(elem) > 1 and first_char == '-' and elem[1].isdigit()):
                                    try:
                                        numeric = int(elem) if '.' not in elem and 'e' not in elem.lower() else float(elem)
                                        member = value_map.get(numeric, _MISSING)
                                        if member is not _MISSING:
                                            converted.append(member)
                                        else:
                                            raise ValueError(f"Invalid enum value: {elem}")
                                    except (ValueError, OverflowError):
                                        raise ValueError(f"Invalid enum value: {elem}")
                                else:
                                    raise ValueError(f"Invalid enum value: {elem}")
                            else:
                                raise ValueError(f"Invalid enum value: {elem}")
                        else:
                            raise ValueError(f"Invalid enum value: {elem}")

                # Preserve container type
                values[field_name] = tuple(converted) if isinstance(value, tuple) else converted
            else:
                # Scalar conversion path - INLINE for performance
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

                    # Optimized numeric string conversion
                    if len(value) > 0:
                        first_char = value[0]
                        if first_char.isdigit() or (len(value) > 1 and first_char == '-' and value[1].isdigit()):
                            try:
                                numeric = int(value) if '.' not in value and 'e' not in value.lower() else float(value)
                                member = value_map.get(numeric, _MISSING)
                                if member is not _MISSING:
                                    values[field_name] = member
                                    continue
                            except (ValueError, OverflowError):
                                pass

                # Invalid value - immediate error
                valid_names = list(names.keys())
                valid_values = list(filter(lambda v: v is not None, value_map.keys()))
                raise ValueError(
                    f"Invalid value for field {field_name}: {value} "
                    f"Valid names: {valid_names}, valid values: {valid_values}"
                )

        return values

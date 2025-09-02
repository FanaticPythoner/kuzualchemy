from __future__ import annotations

from typing import Any, Union, get_origin, get_args, Type, Dict
from enum import Enum
from pydantic import model_validator

from .constants import ErrorMessages, EnumCacheConstants
from .kuzu_orm import KuzuBaseModel

# Module-level cache for enum lookups: {EnumClass: {name->member, value->member}}
_ENUM_CACHE: Dict[Type[Enum], Dict[str, Dict[Any, Enum]]] = {}


class BaseModel(KuzuBaseModel):
    """
    Base model with automatic enum conversion for Kuzu ORM
    
    :class:`BaseModel`
    :synopsis: Base model with automatic enum conversion for Kuzu ORM
    :platform: Any
    :ivar _ENUM_CACHE: Module-level cache for enum lookups
    """
    
    @staticmethod
    def _get_enum_lookups(enum_type: Type[Enum]) -> Dict[str, Dict[Any, Enum]]:
        """
        Get or create cached lookup dictionaries for an enum type.
        
        :param enum_type: The enum class to create lookups for
        :type enum_type: Type[Enum]
        :returns: Dictionary with 'names' and 'values' mappings for O(1) lookup
        :rtype: Dict[str, Dict[Any, Enum]]
        """
        global _ENUM_CACHE
        if enum_type not in _ENUM_CACHE:
            names = {}
            values = {}
            for member in enum_type:
                names[member.name] = member
                # Store value as string for consistent lookup
                values[str(member.value)] = member
                # Also store numeric values directly if applicable
                if isinstance(member.value, (int, float)):
                    values[member.value] = member
            _ENUM_CACHE[enum_type] = {
                EnumCacheConstants.NAMES_KEY: names,
                EnumCacheConstants.VALUES_KEY: values
            }
        return _ENUM_CACHE[enum_type]
    
    @model_validator(mode='before')
    @classmethod
    def convert_str_to_enum(cls: Type['BaseModel'], values: Any) -> Any:
        """
        Convert string inputs to Enum types if the field is an Enum.
        
        :param cls: The model class
        :type cls: Type['BaseModel']
        :param values: Input values to validate
        :type values: Any
        :returns: Validated values with enums converted
        :rtype: Any
        :raises ValueError: If enum value is invalid
        """
        # @@ STEP 1: Check if values is a dictionary
        # || S.1.1: Use type() instead of isinstance
        if type(values) is not dict:
            return values
            
        # Get model annotations
        annotations = getattr(cls, '__annotations__', {})
        
        # Process each annotated field
        for field_name, field_type in annotations.items():
            # Skip if field not present in input
            if field_name not in values:
                continue
                
            value = values[field_name]
            
            # Skip None values
            if value is None:
                continue
                
            # @@ STEP 2: Process only string values
            # || S.2.1: Use type() instead of isinstance
            if type(value) is not str:
                continue
            
            # Handle Union types (e.g., Optional[EnumType])
            origin = get_origin(field_type)
            if origin is Union:
                # Find the Enum type in the Union args
                args = get_args(field_type)
                enum_type = None
                for arg in args:
                    # @@ STEP 4: Find Enum type in Union args
                    # || S.4.1: Check without isinstance
                    try:
                        if type(arg) is type and issubclass(arg, Enum):
                            enum_type = arg
                            break
                    except TypeError:
                        # || S.4.2: Not a class type - SKIP this arg, not an error
                        continue
                if enum_type:
                    field_type = enum_type
                else:
                    continue
            
            # @@ STEP 3: Check if field_type is an Enum subclass
            # || S.3.1: Check type and subclass without isinstance
            try:
                if not (type(field_type) is type and issubclass(field_type, Enum)):
                    continue
            except TypeError:
                # || S.3.2: Not a class type - SKIP this field, not an error
                continue
                
            # Get cached lookups for O(1) access
            lookups = BaseModel._get_enum_lookups(field_type)
            names = lookups[EnumCacheConstants.NAMES_KEY]
            value_map = lookups[EnumCacheConstants.VALUES_KEY]
            
            # Direct lookup - O(1) for all cases
            # Try member name
            member = names.get(value)
            if member is not None:
                values[field_name] = member
                continue
                
            # Try direct value
            member = value_map.get(value)
            if member is not None:
                values[field_name] = member
                continue
                
            # Try numeric conversion - only if string looks numeric
            # Quick check for digits to avoid expensive conversion attempts
            if value[0].isdigit() or (len(value) > 1 and value[0] == '-' and value[1].isdigit()):
                # Try int
                if '.' not in value and 'e' not in value.lower():
                    numeric = int(value)
                    member = value_map.get(numeric)
                    if member is not None:
                        values[field_name] = member
                        continue
                else:
                    # Try float
                    numeric = float(value)
                    member = value_map.get(numeric)
                    if member is not None:
                        values[field_name] = member
                        continue
            
            # @@ STEP 5: Invalid value - build error message
            # || S.5.1: Filter values without isinstance
            valid_names = list(names.keys())
            valid_values = []
            for k in value_map.keys():
                try:
                    # || S.5.2: Check if k is an Enum member
                    if not issubclass(type(k), Enum):
                        valid_values.append(str(k))
                except TypeError:
                    # Not a class - just append as string
                    valid_values.append(str(k))
            
            # || S.5.3: Use error message from constants
            raise ValueError(
                ErrorMessages.INVALID_FIELD_VALUE.format(
                    field_name=field_name, 
                    value=value
                ) + f" Valid names: {valid_names}, valid values: {valid_values}"
            )
        
        return values
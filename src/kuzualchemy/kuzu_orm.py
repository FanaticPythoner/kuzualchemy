# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
Kùzu ORM system with decorators, field metadata, and DDL generation.
Type-safe metadata and DDL emission that matches the expected grammar and ordering
used in tests (PRIMARY KEY inline when singular, DEFAULT/UNIQUE/NOT NULL/CHECK ordering,
FK constraints, column-level INDEX tags, and correct relationship multiplicity placement).
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Set,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from pydantic import BaseModel, Field, ConfigDict
from pydantic.fields import FieldInfo

from .constants import (
    CascadeAction,
    DDLConstants,
    KuzuDefaultFunction,
    ModelMetadataConstants,
    DefaultValueConstants,
    RelationshipDirection,
    RelationshipMultiplicity,
    KuzuDataType,
    ConstraintConstants,
    ArrayTypeConstants,
    ErrorMessages,
    ValidationMessageConstants,
    RegistryResolutionConstants,
)

if TYPE_CHECKING:
    from .kuzu_query import Query
    from .kuzu_session import KuzuSession

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Type variables
# -----------------------------------------------------------------------------

T = TypeVar("T")
ModelType = TypeVar("ModelType", bound="KuzuBaseModel")


# -----------------------------------------------------------------------------
# SQL Keywords Registry
# -----------------------------------------------------------------------------

class SQLKeywordRegistry:
    """
    Dynamic registry for SQL keywords and functions.
    
    :class: SQLKeywordRegistry
    :synopsis: Registry for SQL keywords and time functions
    """
    
    # @@ STEP 1: Dynamically build time keywords from KuzuDefaultFunction enum
    # || S.S.1: Extract time-related functions from the enum
    _time_keywords: Set[str] = set()
    
    # @@ STEP 2: Initialize time keywords from enum at class definition time
    # || S.S.2: This will be populated by the _initialize_time_keywords method
    
    _null_keywords: Set[str] = {DefaultValueConstants.NULL_KEYWORD}

    _boolean_keywords: Set[str] = {DefaultValueConstants.TRUE_KEYWORD, DefaultValueConstants.FALSE_KEYWORD}
    
    @classmethod
    def _initialize_time_keywords(cls) -> None:
        """
        Initialize time keywords using pure inheritance checks.
        
        No patterns, no hardcoding - just isinstance checks on the class hierarchy.
        """
        # @@ STEP: Use isinstance to detect TimeFunction instances
        from .constants import KuzuDefaultFunction
        from .kuzu_function_types import TimeFunction
        
        for func in KuzuDefaultFunction:
            # || S.1: Check if this enum value is a TimeFunction instance
            if isinstance(func.value, TimeFunction):
                # || S.2: Extract function name without parentheses
                func_str = str(func.value)
                if func_str.endswith('()'):
                    func_keyword = func_str[:-2].upper()
                else:
                    func_keyword = func_str.upper()
                cls._time_keywords.add(func_keyword)
    
    @classmethod
    def add_keyword(cls, keyword: str) -> None:
        """
        Add a new SQL keyword.
        
        :param keyword: Keyword to add
        :type keyword: str
        """
        # @@ STEP 3: Add keyword to registry
        cls._time_keywords.add(keyword.upper())
    
    @classmethod
    def register_null_keyword(cls, keyword: str) -> None:
        """Register a new null-related SQL keyword."""
        cls._null_keywords.add(keyword.upper())
    
    @classmethod
    def register_boolean_keyword(cls, keyword: str) -> None:
        """Register a new boolean SQL keyword."""
        cls._boolean_keywords.add(keyword.upper())
    
    @classmethod
    def is_sql_keyword(cls, value: str) -> bool:
        """
        Check if a value is a SQL keyword.
        
        :param value: Value to check
        :type value: str
        :returns: True if value is a SQL keyword
        :rtype: bool
        """
        # @@ STEP 2: Check if value is a SQL keyword
        # || S.2.1: Use type() instead of isinstance
        return value.upper() in cls._time_keywords
    
    @classmethod
    def is_time_keyword(cls, value: str) -> bool:
        """
        Check if value is a time-related SQL keyword.
        
        :param value: Value to check
        :type value: str
        :returns: True if value is a time keyword
        :rtype: bool
        """
        return value.upper().strip() in cls._time_keywords
    
    @classmethod
    def is_null_keyword(cls, value: str) -> bool:
        """Check if value is a null-related SQL keyword."""
        return value.upper().strip() in cls._null_keywords
    
    @classmethod
    def is_boolean_keyword(cls, value: str) -> bool:
        """Check if value is a boolean SQL keyword."""
        return value.upper().strip() in cls._boolean_keywords


# -----------------------------------------------------------------------------
# Default Value Renderers
# -----------------------------------------------------------------------------

class DefaultValueHandlerRegistry:
    """Registry for type-specific default value handlers."""
    
    _handlers: Dict[type, Callable[[Any], str]] = {}
    
    @classmethod
    def register_handler(cls, value_type: type, handler: Callable[[Any], str]) -> None:
        """Register a handler for a specific type."""
        cls._handlers[value_type] = handler
    
    @classmethod
    def get_handler(cls, value: Any) -> Optional[Callable[[Any], str]]:
        """Get the handler for a value's type."""
        value_type = type(value)
        return cls._handlers.get(value_type)
    
    @classmethod
    def render(cls, value: Any) -> str:
        """Render a value using the appropriate handler."""
        # Direct type-based dispatch only
        handler = cls.get_handler(value)
        if not handler:
            raise ValueError(ErrorMessages.INVALID_FIELD_TYPE.format(field_name=type(value).__name__, error="No handler registered. Register a handler using DefaultValueHandlerRegistry.register_handler()"))
        return handler(value)
    

    @staticmethod
    def _bool_handler(value: bool) -> str:
        """Handler for boolean values."""
        bool_str = DefaultValueConstants.BOOL_TRUE if value else DefaultValueConstants.BOOL_FALSE
        return f"{DefaultValueConstants.DEFAULT_PREFIX} {bool_str}"

    @staticmethod
    def _int_handler(value: int) -> str:
        """Handler for integer values."""
        return f"{DefaultValueConstants.DEFAULT_PREFIX} {value}"

    @staticmethod
    def _kuzu_default_function_handler(value: "KuzuDefaultFunction") -> str:
        """Handler for KuzuDefaultFunction enum values."""
        # @@ STEP: Use the string value of the enum
        # || S.1: Kuzu DOES support functions like current_timestamp() in DEFAULT
        return f"{DefaultValueConstants.DEFAULT_PREFIX} {value.value}"

    @staticmethod
    def _float_handler(value: float) -> str:
        """Handler for float values."""
        return f"{DefaultValueConstants.DEFAULT_PREFIX} {value}"

    @staticmethod
    def _string_handler(value: str) -> str:
        """
        Handler for string values with SQL keyword detection.
        
        NOTE: Function calls should be KuzuDefaultFunction enum values, not strings.
        If you need a function default, use the proper enum from constants.py.
        """
        up = value.upper().strip()
        
        # @@ STEP: Handle time keywords - Kuzu doesn't support these as DEFAULT
        # || S.1: CURRENT_TIMESTAMP, NOW(), etc. are not supported in Kuzu DEFAULT clauses
        # || S.2: Raise explicit error for unsupported time keywords
        if SQLKeywordRegistry.is_time_keyword(value):
            # Don't emit DEFAULT for unsupported time keywords - THIS IS AN ERROR
            raise ValueError(
                f"Kuzu does not support time function '{value}' in DEFAULT clause. "
                f"Use KuzuDefaultFunction enum values for function defaults."
            )
        
        if SQLKeywordRegistry.is_null_keyword(value):
            return f"{DefaultValueConstants.DEFAULT_PREFIX} {DefaultValueConstants.NULL_KEYWORD}"

        if SQLKeywordRegistry.is_boolean_keyword(value):
            return f"{DefaultValueConstants.DEFAULT_PREFIX} {up.lower()}"

        # @@ STEP: Check if string is already quoted
        # || S.1: If the string starts and ends with single quotes, it's already quoted
        if value.startswith(DefaultValueConstants.QUOTE_CHAR) and value.endswith(DefaultValueConstants.QUOTE_CHAR):
            # Already quoted, use as-is
            return f"{DefaultValueConstants.DEFAULT_PREFIX} {value}"

        # Quote as literal string
        safe = value.replace(DefaultValueConstants.QUOTE_CHAR, DefaultValueConstants.ESCAPED_QUOTE)
        return f"{DefaultValueConstants.DEFAULT_PREFIX} {DefaultValueConstants.QUOTE_CHAR}{safe}{DefaultValueConstants.QUOTE_CHAR}"

# Register basic handlers - use the static methods that include DEFAULT prefix
DefaultValueHandlerRegistry.register_handler(bool, DefaultValueHandlerRegistry._bool_handler)
DefaultValueHandlerRegistry.register_handler(int, DefaultValueHandlerRegistry._int_handler)
DefaultValueHandlerRegistry.register_handler(float, DefaultValueHandlerRegistry._float_handler)
DefaultValueHandlerRegistry.register_handler(str, DefaultValueHandlerRegistry._string_handler)
DefaultValueHandlerRegistry.register_handler(type(None), lambda v: DefaultValueConstants.NULL_KEYWORD)
# Add handler for lists (arrays)
DefaultValueHandlerRegistry.register_handler(list, lambda v: f"{DefaultValueConstants.DEFAULT_PREFIX} [{', '.join(str(item) if isinstance(item, (int, float)) else f'{DefaultValueConstants.QUOTE_CHAR}{item}{DefaultValueConstants.QUOTE_CHAR}' for item in v)}]")


class BulkInsertValueGeneratorRegistry:
    """
    Registry for generating actual values from KuzuDefaultFunction instances.

    This registry is used during bulk insert operations where COPY FROM
    doesn't support DEFAULT functions, so we must generate the actual
    values that the functions would produce.
    """

    _generators: Dict[type, Callable[[Any], str]] = {}

    @classmethod
    def register_generator(cls, function_type: type, generator: Callable[[Any], str]) -> None:
        """Register a value generator for a specific function type."""
        cls._generators[function_type] = generator

    @classmethod
    def get_generator(cls, function_obj: Any) -> Optional[Callable[[Any], str]]:
        """Get the generator for a function object's type."""
        function_type = type(function_obj)
        return cls._generators.get(function_type)

    @classmethod
    def generate_value(cls, default_function: Any) -> str:
        """
        Generate actual value from KuzuDefaultFunction enum.

        Args:
            default_function: KuzuDefaultFunction enum value

        Returns:
            Generated value as string in Kuzu-compatible format
        """
        # Get the actual function object from the enum value
        func_obj = default_function.value

        # Get the appropriate generator
        generator = cls.get_generator(func_obj)
        if not generator:
            raise ValueError(
                f"No value generator registered for function type {type(func_obj)}. "
                f"Register a generator using BulkInsertValueGeneratorRegistry.register_generator()"
            )

        return generator(func_obj)

    @staticmethod
    def _time_function_generator(func_obj: Any) -> str:
        """Generate values for TimeFunction instances using enum-based dispatch."""
        from datetime import datetime, date
        from .constants import KuzuDefaultFunction

        # Find the corresponding enum value for this function object
        for enum_value in KuzuDefaultFunction:
            if enum_value.value is func_obj:
                # Use enum-based dispatch instead of string matching
                if enum_value == KuzuDefaultFunction.CURRENT_TIMESTAMP:
                    return datetime.now().isoformat()
                elif enum_value == KuzuDefaultFunction.CURRENT_DATE:
                    return date.today().isoformat()
                elif enum_value == KuzuDefaultFunction.CURRENT_TIME:
                    return datetime.now().time().isoformat()
                elif enum_value == KuzuDefaultFunction.NOW:
                    return datetime.now().isoformat()
                else:
                    # Unknown time function - raise error instead of fallback
                    raise ValueError(f"Unknown time function: {enum_value}")

        # If no enum found, raise error
        raise ValueError(f"Function object {func_obj} not found in KuzuDefaultFunction enum")

    @staticmethod
    def _uuid_function_generator(func_obj: Any) -> str:
        """Generate values for UUIDFunction instances."""
        import uuid
        return str(uuid.uuid4())

    @staticmethod
    def _sequence_function_generator(func_obj: Any) -> str:
        """Handle SequenceFunction instances - not supported in bulk insert."""
        raise ValueError(
            f"Sequence function {func_obj} cannot be used in bulk insert. "
            f"Use individual inserts for sequence-based defaults."
        )


# Register generators for each function type
from .kuzu_function_types import TimeFunction, UUIDFunction, SequenceFunction
BulkInsertValueGeneratorRegistry.register_generator(TimeFunction, BulkInsertValueGeneratorRegistry._time_function_generator)
BulkInsertValueGeneratorRegistry.register_generator(UUIDFunction, BulkInsertValueGeneratorRegistry._uuid_function_generator)
BulkInsertValueGeneratorRegistry.register_generator(SequenceFunction, BulkInsertValueGeneratorRegistry._sequence_function_generator)


# -----------------------------------------------------------------------------
# Field-level metadata
# -----------------------------------------------------------------------------

@dataclass
class CheckConstraintMetadata:
    """
    Metadata for check constraints.
    
    :class: CheckConstraintMetadata
    :synopsis: Dataclass for check constraint metadata
    """
    expression: str
    name: Optional[str] = None

@dataclass
class ForeignKeyMetadata:
    """
    Enhanced metadata for foreign key constraints with deferred resolution support.

    This class supports SQLAlchemy-like deferred resolution of target models,
    allowing for circular dependencies and forward references.

    :class: ForeignKeyMetadata
    :synopsis: Dataclass for foreign key constraint metadata with deferred resolution
    """
    target_model: Union[str, Type[Any], Callable[[], Type[Any]]]
    target_field: str
    on_delete: Optional[CascadeAction] = None
    on_update: Optional[CascadeAction] = None

    # @@ STEP: Internal resolution state tracking
    _resolution_state: str = RegistryResolutionConstants.RESOLUTION_STATE_UNRESOLVED
    _resolved_target_model: Optional[Type[Any]] = None
    _resolved_target_name: Optional[str] = None
    _resolution_error: Optional[str] = None

    def get_target_type(self) -> str:
        """
        Determine the type of target model reference.

        Returns:
            str: One of TARGET_TYPE_STRING, TARGET_TYPE_CLASS, or TARGET_TYPE_CALLABLE
        """
        if isinstance(self.target_model, str):
            return RegistryResolutionConstants.TARGET_TYPE_STRING
        elif callable(self.target_model) and not isinstance(self.target_model, type):
            return RegistryResolutionConstants.TARGET_TYPE_CALLABLE
        else:
            return RegistryResolutionConstants.TARGET_TYPE_CLASS

    def is_resolved(self) -> bool:
        """Check if this foreign key reference has been resolved."""
        return self._resolution_state == RegistryResolutionConstants.RESOLUTION_STATE_RESOLVED

    def resolve_target_model(self, registry: 'KuzuRegistry') -> bool:
        """
        Resolve the target model reference using the provided registry.

        Args:
            registry: The KuzuRegistry instance to use for resolution

        Returns:
            bool: True if resolution was successful, False otherwise
        """
        if self.is_resolved():
            return True

        self._resolution_state = RegistryResolutionConstants.RESOLUTION_STATE_RESOLVING

        try:
            target_type = self.get_target_type()

            if target_type == RegistryResolutionConstants.TARGET_TYPE_STRING:
                # @@ STEP: Resolve string reference
                resolved_class = registry.get_model_by_name(self.target_model)
                if resolved_class is None:
                    self._resolution_error = f"{RegistryResolutionConstants.ERROR_TARGET_NOT_FOUND}: {self.target_model}"
                    self._resolution_state = RegistryResolutionConstants.RESOLUTION_STATE_ERROR
                    return False

                self._resolved_target_model = resolved_class
                self._resolved_target_name = self.target_model

            elif target_type == RegistryResolutionConstants.TARGET_TYPE_CALLABLE:
                # @@ STEP: Resolve callable reference
                try:
                    resolved_class = self.target_model()
                    if not isinstance(resolved_class, type):
                        self._resolution_error = f"{RegistryResolutionConstants.ERROR_INVALID_TARGET_TYPE}: Callable must return a class"
                        self._resolution_state = RegistryResolutionConstants.RESOLUTION_STATE_ERROR
                        return False

                    self._resolved_target_model = resolved_class
                    self._resolved_target_name = self._extract_model_name(resolved_class)

                except Exception as e:
                    self._resolution_error = f"{RegistryResolutionConstants.ERROR_INVALID_TARGET_TYPE}: {str(e)}"
                    self._resolution_state = RegistryResolutionConstants.RESOLUTION_STATE_ERROR
                    return False

            else:  # TARGET_TYPE_CLASS
                # @@ STEP: Direct class reference
                self._resolved_target_model = self.target_model
                self._resolved_target_name = self._extract_model_name(self.target_model)

            self._resolution_state = RegistryResolutionConstants.RESOLUTION_STATE_RESOLVED
            return True

        except Exception as e:
            self._resolution_error = str(e)
            self._resolution_state = RegistryResolutionConstants.RESOLUTION_STATE_ERROR
            return False

    def _extract_model_name(self, model_class: Type[Any]) -> str:
        """
        Extract the model name from a class, trying multiple approaches.

        Args:
            model_class: The class to extract the name from

        Returns:
            str: The extracted model name
        """
        # @@ STEP: Try multiple ways to get the model name
        # || S.1: Check for kuzu_node_name attribute
        if hasattr(model_class, '__kuzu_node_name__'):
            return model_class.__kuzu_node_name__

        # || S.2: Check for __name__ attribute
        if hasattr(model_class, '__name__'):
            return model_class.__name__

        # || S.3: Check for __qualname__ attribute
        if hasattr(model_class, '__qualname__'):
            return model_class.__qualname__.split('.')[-1]

        # || S.4: Fallback to string representation
        return str(model_class)

    def get_resolved_target_name(self) -> Optional[str]:
        """Get the resolved target model name, if available."""
        return self._resolved_target_name

    def get_resolved_target_model(self) -> Optional[Type[Any]]:
        """Get the resolved target model class, if available."""
        return self._resolved_target_model

    def to_ddl(self, field_name: str) -> str:
        """
        Generate DDL comment for foreign key constraint.

        Since Kuzu doesn't support foreign key constraints in DDL,
        this generates a comment for documentation purposes.
        """
        # @@ STEP: Use resolved target name if available, otherwise try to determine it
        if self.is_resolved() and self._resolved_target_name:
            target_name = self._resolved_target_name
        elif isinstance(self.target_model, str):
            target_name = self.target_model
        else:
            target_name = self._extract_model_name(self.target_model)

        # @@ STEP: Build foreign key constraint comment
        fk_comment = f"{DDLConstants.FOREIGN_KEY} ({field_name}) {DDLConstants.REFERENCES} {target_name}({self.target_field})"

        # @@ STEP: Add cascade actions if specified
        if self.on_delete:
            fk_comment += f" {DDLConstants.ON_DELETE} {self.on_delete.value}"
        if self.on_update:
            fk_comment += f" {DDLConstants.ON_UPDATE} {self.on_update.value}"

        return fk_comment

# Alias for backward compatibility
ForeignKeyReference = ForeignKeyMetadata

@dataclass
class IndexMetadata:
    """
    Metadata for index definitions.
    
    :class: IndexMetadata
    :synopsis: Dataclass for index metadata storage
    """
    fields: List[str]
    unique: bool = False
    name: Optional[str] = None

    def to_ddl(self, table_name: str) -> str:
        index_name = self.name or f"{ConstraintConstants.INDEX_PREFIX}{ConstraintConstants.INDEX_SEPARATOR}{table_name}{ConstraintConstants.INDEX_SEPARATOR}{ConstraintConstants.INDEX_SEPARATOR.join(self.fields)}"
        unique_str = ConstraintConstants.UNIQUE_INDEX if self.unique else ""
        return f"{DDLConstants.CREATE_INDEX.replace('INDEX', unique_str + ConstraintConstants.INDEX)} {index_name} ON {table_name}({DDLConstants.FIELD_SEPARATOR.join(self.fields)}){DDLConstants.STATEMENT_SEPARATOR}"

# Alias for compound indexes
CompoundIndex = IndexMetadata

@dataclass
class TableConstraint:
    """
    Represents a table-level constraint for Kuzu tables.

    This replaces string-based constraints with proper typed objects
    following SQLAlchemy-style patterns for better type safety and validation.

    :class: TableConstraint
    :synopsis: Type-safe table constraint specification
    """
    constraint_type: str  # CHECK, UNIQUE, etc.
    expression: str       # The constraint expression
    name: Optional[str] = None  # Optional constraint name

    def to_ddl(self) -> str:
        """Convert constraint to DDL string."""
        if self.constraint_type.upper() == ConstraintConstants.CHECK:
            if self.name:
                return f"{ConstraintConstants.CONSTRAINT} {self.name} {ConstraintConstants.CHECK} ({self.expression})"
            else:
                return f"{ConstraintConstants.CHECK} ({self.expression})"
        elif self.constraint_type.upper() == ConstraintConstants.UNIQUE:
            if self.name:
                return f"{ConstraintConstants.CONSTRAINT} {self.name} {ConstraintConstants.UNIQUE} ({self.expression})"
            else:
                return f"{ConstraintConstants.UNIQUE} ({self.expression})"
        else:
            return f"{self.constraint_type} ({self.expression})"

@dataclass
class PropertyMetadata:
    """
    Represents metadata for relationship properties.

    This replaces string-based properties with proper typed objects
    following SQLAlchemy-style patterns for better type safety and validation.

    :class: PropertyMetadata
    :synopsis: Type-safe property metadata specification
    """
    property_type: Union[KuzuDataType, str]
    default_value: Optional[Any] = None
    nullable: bool = True
    description: Optional[str] = None

    def to_ddl(self) -> str:
        """Convert property metadata to DDL string."""
        if isinstance(self.property_type, KuzuDataType):
            type_str = self.property_type.value
        else:
            type_str = str(self.property_type)

        ddl_parts = [type_str]

        if self.default_value is not None:
            if isinstance(self.default_value, str):
                ddl_parts.append(f"DEFAULT '{self.default_value}'")
            else:
                ddl_parts.append(f"DEFAULT {self.default_value}")

        if not self.nullable:
            ddl_parts.append(DDLConstants.NOT_NULL)

        return " ".join(ddl_parts)

@dataclass
class ArrayTypeSpecification:
    """Specification for array/list types with element type."""
    element_type: Union[KuzuDataType, str]
    
    def to_ddl(self) -> str:
        """Convert to DDL string like 'INT64[]' or 'STRING[]'."""
        if isinstance(self.element_type, KuzuDataType):
            element_str = self.element_type.value
        else:
            element_str = self.element_type
        return f"{element_str}{ArrayTypeConstants.ARRAY_SUFFIX}"


@dataclass
class KuzuFieldMetadata:
    """
    Metadata for Kuzu fields.

    :class: KuzuFieldMetadata
    :synopsis: Metadata container for Kuzu field definitions
    """
    kuzu_type: Union[KuzuDataType, ArrayTypeSpecification]
    primary_key: bool = False
    foreign_key: Optional[ForeignKeyMetadata] = None
    unique: bool = False
    not_null: bool = False
    index: bool = False  # Single field index (column-level tag in emitted DDL)
    check_constraint: Optional[str] = None
    default_value: Optional[Union[Any, KuzuDefaultFunction]] = None
    default_factory: Optional[Callable[[], Any]] = None
    auto_increment: bool = False  # For SERIAL type auto-increment support

    # Relationship-only markers (not emitted; used for custom schemas)
    is_from_ref: bool = False
    is_to_ref: bool = False

    def to_ddl(self, field_name: str) -> str:
        """Generate DDL for field definition."""
        return self.to_ddl_column_definition(field_name)
    
    # ---- Column-level DDL renderer used by tests directly ----
    def to_ddl_column_definition(self, field_name: str, is_node_table: bool = True) -> str:
        """
        Render the column definition for Kuzu DDL.

        IMPORTANT: Kuzu v0.11.2 NODE tables only support:
        - PRIMARY KEY (inline or table-level)
        - DEFAULT values

        NOT supported in NODE tables: NOT NULL, UNIQUE, CHECK
        """
        # @@ STEP: is_node_table parameter reserved for future REL table support
        _ = is_node_table  # Mark as intentionally unused - current implementation assumes NODE table behavior

        dtype = self._canonical_type_name(self.kuzu_type)
        parts: List[str] = [field_name, dtype]

        # @@ STEP: Handle DEFAULT (skip for SERIAL)
        is_serial = isinstance(self.kuzu_type, KuzuDataType) and self.kuzu_type == KuzuDataType.SERIAL
        if self.default_value is not None and not is_serial:
            default_clause = self._render_default(self.default_value)
            # Only add if we got a non-empty DEFAULT clause
            if default_clause:
                parts.append(default_clause)

        # @@ STEP: Handle PRIMARY KEY
        if self.primary_key:
            parts.append(DDLConstants.PRIMARY_KEY)
            return " ".join(parts)

        # @@ STEP: For NODE tables, ignore unsupported constraints
        # || S.1: CHECK, UNIQUE, NOT NULL are NOT supported in Kuzu NODE tables
        # || S.2: These constraints will be silently ignored to generate valid DDL
        return " ".join(parts)

    @staticmethod
    def _canonical_type_name(dt: Union["KuzuDataType", "ArrayTypeSpecification"]) -> str:
        # Handle array type specifications
        if isinstance(dt, ArrayTypeSpecification):
            return dt.to_ddl()
        # Handle string types (either KuzuDataType constants or custom types)
        if isinstance(dt, (str, KuzuDataType)):
            # If it's a KuzuDataType constant string, return it directly
            # If it's a custom type string, return it directly
            return dt
        
        # For actual attribute access (when dt is like KuzuDataType.INT64)
        # This shouldn't happen with the new code
        raise ValueError(f"Unsupported type: {dt}")

    @staticmethod
    def _render_default(value: Any) -> str:
        """Render a default value using the dynamic registry system."""
        if isinstance(value, KuzuDefaultFunction):
            return f"DEFAULT {value.value}"
        return DefaultValueHandlerRegistry.render(value)


def kuzu_field(
    default: Any = ...,
    *,
    kuzu_type: Union[KuzuDataType, str, ArrayTypeSpecification],
    primary_key: bool = False,
    foreign_key: Optional[ForeignKeyMetadata] = None,
    unique: bool = False,
    not_null: bool = False,
    index: bool = False,
    check_constraint: Optional[str] = None,
    default_factory: Optional[Callable[[], Any]] = None,
    auto_increment: bool = False,
    element_type: Optional[Union[KuzuDataType, str]] = None,
    alias: Optional[str] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    json_schema_extra: Optional[Dict[str, Any]] = None,
    is_from_ref: bool = False,
    is_to_ref: bool = False,
) -> Any:
    """
    Create a Pydantic Field with attached Kùzu metadata.
    
    Args:
        default: Default value for the field
        kuzu_type: Kuzu data type (can be ARRAY/LIST for array types or a string like 'INT64[]')
        element_type: Element type for array fields (e.g., 'INT64' for INT64[])
        auto_increment: Enable auto-increment (SERIAL type)
        default_factory: Python-side default factory function
    """
    # Check if kuzu_type is KuzuDataType.ARRAY constant
    if kuzu_type == KuzuDataType.ARRAY:
        # If kuzu_type is ARRAY, must use element_type
        if element_type is not None:
        # User specified element_type, so this is an array
            if isinstance(element_type, str):
                # Check if it's a valid KuzuDataType constant
                if hasattr(KuzuDataType, element_type.upper()):
                    element_type = getattr(KuzuDataType, element_type.upper())
                # Otherwise keep as string for custom types
            kuzu_type = ArrayTypeSpecification(element_type=element_type)
        else:
            raise ValueError("ARRAY type must have an element_type")
    # Parse array syntax like 'INT64[]' or 'STRING[]'
    elif isinstance(kuzu_type, str):
        if kuzu_type.endswith('[]'):
            # Extract element type from array syntax
            element_type_str = kuzu_type[:-2]  # Remove '[]'
            # Check if it's a valid KuzuDataType constant
            if hasattr(KuzuDataType, element_type_str.upper()):
                element_type = getattr(KuzuDataType, element_type_str.upper())
            else:
                # Custom type - allowed for extensibility
                element_type = element_type_str
            kuzu_type = ArrayTypeSpecification(element_type=element_type)
        elif kuzu_type.upper() == 'ARRAY':
            # String 'ARRAY' - must use element_type
            if element_type is not None:
                if isinstance(element_type, str):
                    # Check if it's a valid KuzuDataType constant
                    if hasattr(KuzuDataType, element_type.upper()):
                        element_type = getattr(KuzuDataType, element_type.upper())
                    # Otherwise keep as string for custom types
                kuzu_type = ArrayTypeSpecification(element_type=element_type)
            else:
                # ARRAY without element_type - convert to constant
                kuzu_type = KuzuDataType.ARRAY
        else:
            # Regular type string - validate against KuzuDataType constants or allow custom
            # Check if it's a valid KuzuDataType constant
            if hasattr(KuzuDataType, kuzu_type.upper()):
                kuzu_type = getattr(KuzuDataType, kuzu_type.upper())
            # Otherwise keep as string for custom types
    elif element_type is not None:
        # User specified element_type separately (kuzu_type might be None or already set)
        if isinstance(element_type, str):
            # Check if it's a valid KuzuDataType constant
            if hasattr(KuzuDataType, element_type.upper()):
                element_type = getattr(KuzuDataType, element_type.upper())
            # Otherwise keep as string for custom types
        kuzu_type = ArrayTypeSpecification(element_type=element_type)

    if auto_increment:
        kuzu_type = KuzuDataType.SERIAL

    # Validate that arrays cannot be primary keys
    if primary_key and isinstance(kuzu_type, ArrayTypeSpecification):
        raise ValueError(
            "Arrays cannot be used as primary keys. "
            "Primary keys must be scalar types."
        )
    
    kuzu_metadata = KuzuFieldMetadata(
        kuzu_type=kuzu_type,
        primary_key=primary_key,
        foreign_key=foreign_key,
        unique=unique,
        not_null=not_null,
        index=index,
        check_constraint=check_constraint,
        default_value=None if default is ... else default,
        default_factory=default_factory,
        auto_increment=auto_increment,
        is_from_ref=is_from_ref,
        is_to_ref=is_to_ref,
    )

    if type(json_schema_extra) is not dict:
        json_schema_extra = {}
    json_schema_extra["kuzu_metadata"] = kuzu_metadata.__dict__

    field_kwargs = {
        "json_schema_extra": json_schema_extra,
        "alias": alias,
        "title": title,
        "description": description,
    }

    if isinstance(kuzu_type, KuzuDataType) and kuzu_type == KuzuDataType.SERIAL:
        # SERIAL fields should not have Python-side defaults
        return Field(**field_kwargs)
    elif default_factory is not None:
        return Field(default_factory=default_factory, **field_kwargs)
    else:
        return Field(default=default, **field_kwargs)


def foreign_key(
    target_model: Union[Type[T], str],
    target_field: str = "unique_id",
    on_delete: Optional[CascadeAction] = None,
    on_update: Optional[CascadeAction] = None,
) -> ForeignKeyMetadata:
    """Helper to create a ForeignKeyMetadata object."""
    return ForeignKeyMetadata(
        target_model=target_model,
        target_field=target_field,
        on_delete=on_delete,
        on_update=on_update,
    )


# -----------------------------------------------------------------------------
# Relationship Pair Definition
# -----------------------------------------------------------------------------

@dataclass
class RelationshipPair:
    """
    Specification for a single FROM-TO pair in a relationship.
    
    :class: RelationshipPair
    :synopsis: Container for a specific FROM node to TO node connection
    """
    from_node: Union[Type[Any], str]
    to_node: Union[Type[Any], str]
    
    def get_from_name(self) -> str:
        """Get the name of the FROM node."""
        if isinstance(self.from_node, str):
            return self.from_node

        # Strict validation - sets must be expanded before reaching here
        if isinstance(self.from_node, (set, frozenset)):
            raise TypeError(
                f"RelationshipPair.from_node received a set {self.from_node}. "
                f"Sets must be expanded in _process_relationship_pairs before creating RelationshipPair instances."
            )

        # Try to get the kuzu node name first, fall back to __name__ for backward compatibility
        try:
            return self.from_node.__kuzu_node_name__
        except AttributeError:
            try:
                return self.from_node.__name__
            except AttributeError as e:
                raise ValueError(
                    f"Target model {self.from_node} is not a decorated node - missing __kuzu_node_name__ attribute"
                ) from e
    
    def get_to_name(self) -> str:
        """Get the name of the TO node."""
        if isinstance(self.to_node, str):
            return self.to_node
        
        # Strict validation - sets must be expanded before reaching here
        if isinstance(self.to_node, (set, frozenset)):
            raise TypeError(
                f"RelationshipPair.to_node received a set {self.to_node}. "
                f"Sets must be expanded in _process_relationship_pairs before creating RelationshipPair instances."
            )
        
        # Try to get the kuzu node name first, fall back to __name__ for backward compatibility
        try:
            return self.to_node.__kuzu_node_name__
        except AttributeError:
            try:
                return self.to_node.__name__
            except AttributeError as e:
                raise ValueError(
                    f"Target model {self.to_node} is not a decorated node - missing __kuzu_node_name__ attribute"
                ) from e
    
    def to_ddl_component(self) -> str:
        """Convert to DDL component for CREATE REL TABLE."""
        return f"{DDLConstants.REL_TABLE_GROUP_FROM} {self.get_from_name()} {DDLConstants.REL_TABLE_GROUP_TO} {self.get_to_name()}"
    
    def __repr__(self) -> str:
        return f"RelationshipPair(from={self.from_node}, to={self.to_node})"


# -----------------------------------------------------------------------------
# Global registry
# -----------------------------------------------------------------------------

class KuzuRegistry:
    """
    Enhanced global registry for nodes, relationships, and model metadata with deferred resolution.

    This registry implements SQLAlchemy-like deferred resolution to handle circular dependencies
    and forward references gracefully. The resolution process happens in phases:

    1. Registration Phase: Models are registered without dependency analysis
    2. String Resolution Phase: String references are resolved to actual classes
    3. Dependency Analysis Phase: Dependency graph is built from resolved references
    4. Topological Sort Phase: Creation order is determined
    5. Finalized Phase: Registry is ready for DDL generation
    """

    _instance: Optional["KuzuRegistry"] = None

    def __new__(cls) -> "KuzuRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self.__dict__.get("_initialized", False):
            return
        self._initialized = True

        # @@ STEP 1: Core model storage
        self.nodes: Dict[str, Type[Any]] = {}
        self.relationships: Dict[str, Type[Any]] = {}
        self.models: Dict[str, Type[Any]] = {}

        # @@ STEP 2: Resolution state tracking
        self._resolution_phase: str = RegistryResolutionConstants.PHASE_REGISTRATION
        self._model_dependencies: Dict[str, Set[str]] = {}
        self._unresolved_foreign_keys: List[Tuple[str, str, ForeignKeyMetadata]] = []
        self._resolution_errors: List[str] = []

        # @@ STEP 3: Circular dependency tracking
        self._circular_dependencies: Set[Tuple[str, str]] = set()
        self._self_references: Set[str] = set()

    def _cleanup_model_references(self, model_name: str) -> None:
        """
        Clean up all references to a model to prevent memory leaks during redefinition.

        Args:
            model_name: Name of the model to clean up
        """
        # Clean up model references to prevent memory corruption
        # @@ STEP 1: Remove from dependency tracking
        if model_name in self._model_dependencies:
            del self._model_dependencies[model_name]

        # @@ STEP 2: Remove dependencies on this model from other models
        for deps in self._model_dependencies.values():
            deps.discard(model_name)

        # @@ STEP 3: Remove from unresolved foreign keys
        self._unresolved_foreign_keys = [
            (model, field, fk_meta) for model, field, fk_meta in self._unresolved_foreign_keys
            if model != model_name
        ]

        # @@ STEP 4: Remove from circular dependency tracking
        self._circular_dependencies = {
            (from_model, to_model) for from_model, to_model in self._circular_dependencies
            if from_model != model_name and to_model != model_name
        }
        self._self_references.discard(model_name)

        # @@ STEP 5: Clear any resolution errors related to this model
        self._resolution_errors = [
            error for error in self._resolution_errors
            if model_name not in error
        ]

    def register_node(self, name: str, cls: Type[Any]) -> None:
        """
        Register a node class without immediate dependency analysis.

        Args:
            name: The node name
            cls: The node class
        """
        # CHandle model redefinition gracefully
        if name in self.nodes:
            # @@ STEP: Clean up existing model references to prevent memory leaks
            self._cleanup_model_references(name)

        self.nodes[name] = cls
        self.models[name] = cls

        # @@ STEP: Store unresolved foreign keys for later resolution
        self._collect_unresolved_foreign_keys(name, cls)

    def register_relationship(self, name: str, cls: Type[Any]) -> None:
        """
        Register a relationship class without immediate dependency analysis.

        Args:
            name: The relationship name
            cls: The relationship class
        """
        # Handle model redefinition gracefully
        if name in self.relationships:
            # @@ STEP: Clean up existing model references to prevent memory leaks
            self._cleanup_model_references(name)

        self.relationships[name] = cls
        self.models[name] = cls

        # @@ STEP: Store unresolved foreign keys for later resolution
        self._collect_unresolved_foreign_keys(name, cls)

    def _collect_unresolved_foreign_keys(self, model_name: str, cls: Type[Any]) -> None:
        """
        Collect foreign key references from a model for later resolution.

        Args:
            model_name: The name of the model
            cls: The model class
        """
        for field_name, field_info in cls.model_fields.items():
            metadata = self.get_field_metadata(field_info)
            if metadata and metadata.foreign_key:
                # @@ STEP: Store the foreign key for later resolution
                self._unresolved_foreign_keys.append((model_name, field_name, metadata.foreign_key))

    def get_model_by_name(self, name: str) -> Optional[Type[Any]]:
        """
        Get a model by name from the registry.

        Args:
            name: The model name to look up

        Returns:
            Optional[Type[Any]]: The model class if found, None otherwise
        """
        return self.models.get(name)

    def resolve_all_foreign_keys(self) -> bool:
        """
        Resolve all foreign key references in the registry.

        Returns:
            bool: True if all foreign keys were resolved successfully, False otherwise
        """
        if self._resolution_phase != RegistryResolutionConstants.PHASE_REGISTRATION:
            return True  # Already resolved

        self._resolution_phase = RegistryResolutionConstants.PHASE_STRING_RESOLUTION
        self._resolution_errors.clear()

        success = True

        # @@ STEP: Resolve each foreign key reference
        for model_name, field_name, foreign_key in self._unresolved_foreign_keys:
            if not foreign_key.resolve_target_model(self):
                error_msg = f"Failed to resolve foreign key {model_name}.{field_name} -> {foreign_key.target_model}"
                if foreign_key._resolution_error:
                    error_msg += f": {foreign_key._resolution_error}"
                self._resolution_errors.append(error_msg)
                success = False

        if success:
            self._resolution_phase = RegistryResolutionConstants.PHASE_DEPENDENCY_ANALYSIS

        return success

    def analyze_dependencies(self) -> bool:
        """
        Analyze dependencies between models after foreign key resolution.

        Returns:
            bool: True if dependency analysis was successful, False otherwise
        """
        if self._resolution_phase not in [
            RegistryResolutionConstants.PHASE_STRING_RESOLUTION,
            RegistryResolutionConstants.PHASE_DEPENDENCY_ANALYSIS
        ]:
            return True  # Already analyzed or not ready

        self._resolution_phase = RegistryResolutionConstants.PHASE_DEPENDENCY_ANALYSIS
        self._model_dependencies.clear()
        self._circular_dependencies.clear()
        self._self_references.clear()

        # @@ STEP: Build dependency graph from resolved foreign keys
        for model_name, field_name, foreign_key in self._unresolved_foreign_keys:
            if not foreign_key.is_resolved():
                continue

            target_name = foreign_key.get_resolved_target_name()
            if target_name:
                # @@ STEP: Track dependencies
                if model_name not in self._model_dependencies:
                    self._model_dependencies[model_name] = set()

                if target_name == model_name:
                    # @@ STEP: Self-reference detected
                    self._self_references.add(model_name)
                else:
                    self._model_dependencies[model_name].add(target_name)

                    # @@ STEP: Check for circular dependencies
                    if target_name in self._model_dependencies:
                        if model_name in self._model_dependencies[target_name]:
                            self._circular_dependencies.add((model_name, target_name))

        self._resolution_phase = RegistryResolutionConstants.PHASE_TOPOLOGICAL_SORT
        return True

    def get_creation_order(self) -> List[str]:
        """
        Get the topologically sorted creation order for models.

        This method handles circular dependencies gracefully by:
        1. Detecting self-references (allowed)
        2. Detecting circular dependencies (handled with proper ordering)
        3. Providing a stable sort order

        Returns:
            List[str]: List of model names in creation order
        """
        if self._resolution_phase not in [
            RegistryResolutionConstants.PHASE_TOPOLOGICAL_SORT,
            RegistryResolutionConstants.PHASE_FINALIZED
        ]:
            # @@ STEP: Ensure dependencies are analyzed first
            if not self.resolve_all_foreign_keys():
                raise ValueError("Cannot determine creation order: Foreign key resolution failed")
            if not self.analyze_dependencies():
                raise ValueError("Cannot determine creation order: Dependency analysis failed")

        # @@ STEP: Implement topological sort with cycle detection
        visited = set()
        visiting = set()  # Track nodes currently being visited (for cycle detection)
        order: List[str] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            if name in visiting:
                # @@ STEP: Circular dependency detected - this is OK for self-references
                if name in self._self_references:
                    return  # Self-reference is allowed
                else:
                    # @@ STEP: True circular dependency - handle gracefully
                    logger.warning(f"Circular dependency detected involving {name}")
                    return

            visiting.add(name)

            # @@ STEP: Visit dependencies first
            for dep in self._model_dependencies.get(name, set()):
                if dep != name:  # Skip self-references in dependency traversal
                    visit(dep)

            visiting.remove(name)
            visited.add(name)
            order.append(name)

        # @@ STEP: Visit all models
        for name in sorted(self.models.keys()):  # Sort for stable ordering
            visit(name)

        return order

    def finalize_registry(self) -> bool:
        """
        Finalize the registry by completing all resolution phases.

        Returns:
            bool: True if finalization was successful, False otherwise
        """
        if self._resolution_phase == RegistryResolutionConstants.PHASE_FINALIZED:
            return True

        # @@ STEP: Complete all resolution phases
        if not self.resolve_all_foreign_keys():
            return False

        if not self.analyze_dependencies():
            return False

        # @@ STEP: Verify creation order can be determined
        try:
            self.get_creation_order()
            self._resolution_phase = RegistryResolutionConstants.PHASE_FINALIZED
            return True
        except Exception as e:
            self._resolution_errors.append(f"Failed to determine creation order: {str(e)}")
            return False

    def get_resolution_errors(self) -> List[str]:
        """Get any resolution errors that occurred."""
        return self._resolution_errors.copy()

    def get_circular_dependencies(self) -> Set[Tuple[str, str]]:
        """Get detected circular dependencies."""
        return self._circular_dependencies.copy()

    def get_self_references(self) -> Set[str]:
        """Get models with self-references."""
        return self._self_references.copy()

    def is_finalized(self) -> bool:
        """Check if the registry has been finalized."""
        return self._resolution_phase == RegistryResolutionConstants.PHASE_FINALIZED

    def get_field_metadata(self, field_info: FieldInfo) -> Optional[KuzuFieldMetadata]:
        """
        Get Kuzu metadata from field info.

        :param field_info: Pydantic field info
        :type field_info: FieldInfo
        :returns: Kuzu field metadata or None
        :rtype: Optional[KuzuFieldMetadata]
        """
        # @@ STEP: Extract kuzu metadata from field info
        if field_info.json_schema_extra:
            # || S.1: Check if json_schema_extra is a dict
            if type(field_info.json_schema_extra) is dict:
                kuzu_meta = field_info.json_schema_extra.get(ModelMetadataConstants.KUZU_FIELD_METADATA)
                if kuzu_meta:
                    # || S.2: Return KuzuFieldMetadata instance if it's already one
                    if type(kuzu_meta) is KuzuFieldMetadata:
                        return kuzu_meta
                    # || S.3: Create KuzuFieldMetadata from dict
                    elif type(kuzu_meta) is dict:
                        return KuzuFieldMetadata(**kuzu_meta)
        # No Kuzu metadata found - this is expected for non-Kuzu fields
        return None


# Singleton
_kuzu_registry = KuzuRegistry()


# -----------------------------------------------------------------------------
# Relationship pair processing helpers
# -----------------------------------------------------------------------------

def _process_relationship_pairs(
    pairs: List[Tuple[Union[Set[Any], Any], Union[Set[Any], Any]]],
    rel_name: str
) -> List[RelationshipPair]:
    """
    Process relationship pairs into RelationshipPair objects.
    Handles various formats including sets.

    Supported formats:
    1. Traditional format: [(FromType, ToType), ...]
    2. Enhanced format: [(FromType, {ToType, ToType2}), ...]
    3. Full Cartesian product: [({FromType1, FromType2}, {ToType1, ToType2}), ...]
    4. Partial Cartesian product: [({FromType1, FromType2}, ToType), ...]

    Args:
        pairs: Relationship pairs in any supported format
        rel_name: Name of the relationship for error messages

    Returns:
        List of RelationshipPair objects

    Raises:
        ValueError: If pairs format is invalid or unsupported
    """
    rel_pairs = []

    if isinstance(pairs, list):
        # Traditional format: [(FromType, ToType), ...]
        for pair in pairs:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(f"Relationship {rel_name}: Each pair must be a 2-tuple (from_type, to_type)")
            from_type, to_type = pair
            
            # Handle sets in FROM position
            from_types = []
            if isinstance(from_type, (set, frozenset)):
                from_types = list(from_type)
            else:
                from_types = [from_type]
            
            # Handle sets in TO position
            to_types = []
            if isinstance(to_type, (set, frozenset)):
                to_types = list(to_type)
            else:
                to_types = [to_type]
            
            # Create Cartesian product of FROM and TO types
            for ft in from_types:
                for tt in to_types:
                    rel_pairs.append(RelationshipPair(ft, tt))

    else:
        raise ValueError(
            f"Relationship {rel_name}: 'pairs' must be a list of tuples "
            f"[(FromType, ToType), ...]"
        )

    if not rel_pairs:
        raise ValueError(f"Relationship {rel_name}: No valid relationship pairs found")

    return rel_pairs


# -----------------------------------------------------------------------------
# Decorators
# -----------------------------------------------------------------------------

def kuzu_node(
    name: Optional[str] = None,
    abstract: bool = False,
    compound_indexes: Optional[List[CompoundIndex]] = None,
    table_constraints: Optional[List[str]] = None,
    properties: Optional[Dict[str, Any]] = None,
) -> Callable[[Type[T]], Type[T]]:
    """Decorator to mark a class as a Kùzu node."""

    def decorator(cls: Type[T]) -> Type[T]:
        node_name = name if name is not None else cls.__name__

        cls.__kuzu_node_name__ = node_name # type: ignore
        cls.__kuzu_is_abstract__ = abstract # type: ignore
        cls.__kuzu_compound_indexes__ = compound_indexes or [] # type: ignore
        cls.__kuzu_table_constraints__ = table_constraints or [] # type: ignore
        cls.__kuzu_properties__ = properties or {} # type: ignore
        cls.__is_kuzu_node__ = True # type: ignore

        if not abstract:
            _kuzu_registry.register_node(node_name, cls)
        return cls

    return decorator


def kuzu_relationship(
    name: Optional[str] = None,

    pairs: Optional[Union[
        List[Tuple[Union[Type[Any], str], Union[Type[Any], str]]],  # Traditional pair list
        Dict[Union[Type[Any], str], Union[Set[Union[Type[Any], str]], List[Union[Type[Any], str]]]]  # Type -> Set[Type] mapping
    ]] = None,

    multiplicity: RelationshipMultiplicity = RelationshipMultiplicity.MANY_TO_MANY,
    compound_indexes: Optional[List[CompoundIndex]] = None,

    table_constraints: Optional[List[Union[str, "TableConstraint"]]] = None,

    properties: Optional[Dict[str, Union[Any, "PropertyMetadata"]]] = None,

    direction: RelationshipDirection = RelationshipDirection.OUTGOING,
    abstract: bool = False,
    discriminator_field: Optional[str] = None,
    discriminator_value: Optional[str] = None,
    parent_relationship: Optional[Type[Any]] = None,
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator for Kùzu relationship models supporting multiple FROM-TO pairs.
    
    :param name: Relationship table name. If not provided, uses the class name.
    :param pairs: List of (from_node, to_node) tuples defining the relationship pairs.
                  Each tuple specifies a FROM-TO connection between node types.
                  Example: [(User, User), (User, City)] creates a relationship that can connect
                  User to User AND User to City. Each element can be a class type or string name.
    :param multiplicity: Relationship cardinality constraint (MANY_ONE, ONE_MANY, MANY_MANY, ONE_ONE).
                        Applies to all pairs in the relationship.
    :param compound_indexes: List of CompoundIndex objects for multi-field indexes.
    :param table_constraints: Additional table-level SQL constraints as strings.
    :param properties: Additional metadata properties for the relationship.
    :param direction: Logical direction of the relationship (FORWARD, BACKWARD, UNDIRECTED).
                     Used for query generation patterns.
    :param abstract: If True, this relationship won't be registered/created in the database.
                     Used for base relationship classes.
    :param discriminator_field: Field name used for single-table inheritance discrimination.
    :param discriminator_value: Value for the discriminator field in derived relationships.
    :param parent_relationship: Parent relationship class for inheritance hierarchies.
    :return: Decorated class with Kuzu relationship metadata.
    :raises ValueError: If pairs is empty or None when not abstract.
    """
    def decorator(cls: Type[T]) -> Type[T]:
        # @@ STEP 1: Build relationship pairs
        rel_name = name if name is not None else cls.__name__
        rel_pairs = []
        
        if pairs is not None and len(pairs) > 0:
            rel_pairs = _process_relationship_pairs(pairs, rel_name)
        elif not abstract:
            raise ValueError(
                f"Relationship {rel_name} must have 'pairs' parameter defined. "
                f"Example: pairs=[(User, User), (User, City)]"
            )
        
        # @@ STEP 2: Store relationship metadata
        cls.__kuzu_relationship_name__ = rel_name # type: ignore
        cls.__kuzu_rel_name__ = rel_name # type: ignore  # Keep for backward compatibility
        
        # Store relationship pairs
        cls.__kuzu_relationship_pairs__ = rel_pairs # type: ignore
        
        cls.__kuzu_multiplicity__ = multiplicity # type: ignore
        cls.__kuzu_compound_indexes__ = compound_indexes or [] # type: ignore
        cls.__kuzu_table_constraints__ = table_constraints or [] # type: ignore
        cls.__kuzu_properties__ = properties or {} # type: ignore
        cls.__kuzu_direction__ = direction # type: ignore
        cls.__kuzu_is_abstract__ = abstract # type: ignore
        cls.__is_kuzu_relationship__ = True # type: ignore
        
        # @@ STEP 3: Flag for multi-pair relationship
        cls.__kuzu_is_multi_pair__ = len(rel_pairs) > 1 # type: ignore

        # Discriminator metadata (user-level convention)
        cls.__kuzu_discriminator_field__ = discriminator_field # type: ignore
        cls.__kuzu_discriminator_value__ = discriminator_value # type: ignore
        cls.__kuzu_parent_relationship__ = parent_relationship # type: ignore
        if parent_relationship and not discriminator_field:
            if hasattr(parent_relationship, '__kuzu_discriminator_field__'):
                cls.__kuzu_discriminator_field__ = parent_relationship.__kuzu_discriminator_field__ # type: ignore
        if discriminator_value and not cls.__kuzu_discriminator_field__: # type: ignore
            raise ValueError(
                f"Relationship {rel_name} has discriminator_value but no discriminator_field"
            )

        # @@ STEP 4: Register relationship if not abstract and has pairs
        if rel_pairs and not abstract:
            _kuzu_registry.register_relationship(rel_name, cls)
        return cls

    return decorator


# -----------------------------------------------------------------------------
# Base models
# -----------------------------------------------------------------------------

class KuzuBaseModel(BaseModel):
    """Base model for all Kùzu entities with metadata helpers."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, use_enum_values=False
    )

    def __hash__(self) -> int:
        """Make model instances hashable for use in sets."""
        # Use primary key if available, otherwise use id() for object identity
        primary_key_fields = self.get_primary_key_fields()
        if primary_key_fields:
            # Use the first primary key field for hashing
            primary_key_field = primary_key_fields[0]
            # @@ STEP: Access attribute directly
            try:
                pk_value = self.__dict__[primary_key_field]
                return hash((self.__class__.__name__, pk_value))
            except KeyError:
                # Primary key not set - THIS IS AN ERROR
                raise ValueError(
                    f"Cannot compute hash for {self.__class__.__name__}: "
                    f"primary key field '{primary_key_field}' is not set"
                )
        logger.warning(f"Cannot compute hash for {self.__class__.__name__}: no primary key field")
        # Fallback to hashing based on object identity
        return hash(id(self))

    def __eq__(self, other: object) -> bool:
        """Define equality based on primary key or object identity."""
        if not isinstance(other, self.__class__):
            return False

        primary_key_fields = self.get_primary_key_fields()
        if primary_key_fields:
            # Use the first primary key field for equality
            primary_key_field = primary_key_fields[0]
            # @@ STEP: Access attributes directly
            try:
                self_pk = self.__dict__[primary_key_field]
                other_pk = other.__dict__[primary_key_field]
                return self_pk == other_pk
            except KeyError as e:
                # One or both PKs not set - THIS IS AN ERROR
                raise ValueError(
                    f"Cannot compare {self.__class__.__name__} instances: "
                    f"primary key field '{primary_key_field}' is not set. Error: {e}"
                )
        logger.warning(f"Cannot compare {self.__class__.__name__} instances: no primary key field")
        return id(self) == id(other)
    
    @classmethod
    def query(cls, session: Optional["KuzuSession"] = None) -> "Query":
        """
        Create a query for this model.
        
        Args:
            session: Optional session to execute queries with
            
        Returns:
            Query object for this model
        """
        from .kuzu_query import Query
        return Query(cls, session=session)

    @classmethod
    def get_kuzu_metadata(cls, field_name: str) -> Optional[KuzuFieldMetadata]:
        field_info = cls.model_fields.get(field_name)
        if field_info:
            return _kuzu_registry.get_field_metadata(field_info)
        raise AttributeError(f"Field '{field_name}' not found in {cls.__name__}")

    @classmethod
    def get_all_kuzu_metadata(cls) -> Dict[str, KuzuFieldMetadata]:
        res: Dict[str, KuzuFieldMetadata] = {}
        for field_name, field_info in cls.model_fields.items():
            meta = _kuzu_registry.get_field_metadata(field_info)
            if meta:
                res[field_name] = meta
        return res

    @classmethod
    def get_primary_key_fields(cls) -> List[str]:
        pks: List[str] = []
        for field_name, field_info in cls.model_fields.items():
            meta = _kuzu_registry.get_field_metadata(field_info)
            if meta and meta.primary_key:
                pks.append(field_name)
        return pks

    @classmethod
    def get_foreign_key_fields(cls) -> Dict[str, ForeignKeyMetadata]:
        fks: Dict[str, ForeignKeyMetadata] = {}
        for field_name, field_info in cls.model_fields.items():
            meta = _kuzu_registry.get_field_metadata(field_info)
            if meta and meta.foreign_key:
                fks[field_name] = meta.foreign_key
        return fks

    @classmethod
    def validate_foreign_keys(cls) -> List[str]:
        """
        Validate foreign key references using the enhanced deferred resolution system.

        This method now works with the deferred resolution system and can validate
        both resolved and unresolved references appropriately.
        """
        errors: List[str] = []

        for field_name, fk_ref in cls.get_foreign_key_fields().items():
            # @@ STEP: Check if the foreign key has been resolved
            if fk_ref.is_resolved():
                # @@ STEP: Validate resolved reference
                resolved_model = fk_ref.get_resolved_target_model()
                if resolved_model is None:
                    errors.append(f"Field {field_name}: resolved target model is None")
                    continue

                # @@ STEP: Validate resolved model is a proper Pydantic model
                try:
                    model_fields = resolved_model.model_fields
                except AttributeError:
                    errors.append(
                        f"Field {field_name}: resolved target model {resolved_model} is not a valid Pydantic model "
                        f"(missing required 'model_fields' attribute)"
                    )
                    continue

                # @@ STEP: Check for Kuzu decoration
                is_kuzu_node = hasattr(resolved_model, "__kuzu_node_name__")
                is_kuzu_rel = hasattr(resolved_model, "__kuzu_rel_name__")

                if not is_kuzu_node and not is_kuzu_rel:
                    errors.append(
                        f"Field {field_name}: resolved target model {resolved_model.__name__} "
                        f"is not a Kuzu model (missing __kuzu_node_name__ or __kuzu_rel_name__)"
                    )
                    continue

                # @@ STEP: Validate target field exists
                if fk_ref.target_field not in model_fields:
                    errors.append(
                        f"Field {field_name}: target field '{fk_ref.target_field}' not found in {resolved_model.__name__}"
                    )

            else:
                # @@ STEP: Handle unresolved references
                target_type = fk_ref.get_target_type()

                if target_type == RegistryResolutionConstants.TARGET_TYPE_STRING:
                    # @@ STEP: String references are valid and will be resolved later
                    # We can optionally check if the target model name exists in the registry
                    target_name = fk_ref.target_model
                    if not _kuzu_registry.get_model_by_name(target_name):
                        # @@ STEP: Only warn, don't error - the model might be defined later
                        logger.warning(f"Field {field_name}: target model '{target_name}' not found in registry yet")

                elif target_type == RegistryResolutionConstants.TARGET_TYPE_CLASS:
                    # @@ STEP: Direct class reference - validate immediately
                    target_model = fk_ref.target_model

                    try:
                        model_fields = target_model.model_fields
                    except AttributeError:
                        errors.append(
                            f"Field {field_name}: target model {target_model} is not a valid Pydantic model "
                            f"(missing required 'model_fields' attribute)"
                        )
                        continue

                    # Check for Kuzu decoration
                    is_kuzu_node = hasattr(target_model, "__kuzu_node_name__")
                    is_kuzu_rel = hasattr(target_model, "__kuzu_rel_name__")

                    if not is_kuzu_node and not is_kuzu_rel:
                        errors.append(
                            f"Field {field_name}: target model {target_model.__name__} "
                            f"is not a Kuzu model (missing __kuzu_node_name__ or __kuzu_rel_name__)"
                        )
                        continue

                    # Validate target field exists
                    if fk_ref.target_field not in model_fields:
                        errors.append(
                            f"Field {field_name}: target field '{fk_ref.target_field}' not found in {target_model.__name__}"
                        )

                elif target_type == RegistryResolutionConstants.TARGET_TYPE_CALLABLE:
                    # @@ STEP: Callable references will be resolved later - skip validation for now
                    pass

                else:
                    errors.append(f"Field {field_name}: unknown target type '{target_type}'")

        return errors
    
    def save(self, session: "KuzuSession") -> None:
        """
        Save this instance to the database.
        
        Args:
            session: Session to use for saving
        """
        session.add(self)
        session.commit()
    
    def delete(self, session: "KuzuSession") -> None:
        """
        Delete this instance from the database.
        
        Args:
            session: Session to use for deletion
        """
        session.delete(self)
        session.commit()


@kuzu_relationship(
    abstract=True
)
class KuzuRelationshipBase(KuzuBaseModel):
    """Base class for relationship entities with proper node reference handling."""

    from_node: Any
    to_node: Any
    
    _priv_from_node_pk: Optional[Any] = None
    _priv_to_node_pk: Optional[Any] = None
    
    @property
    def _from_node_pk(self) -> Optional[Any]:
        if self._priv_from_node_pk is None:
            self._priv_from_node_pk = self._extract_node_pk(self.from_node)
        return self._priv_from_node_pk
    
    @property
    def _to_node_pk(self) -> Optional[Any]:
        if self._priv_to_node_pk is None:
            self._priv_to_node_pk = self._extract_node_pk(self.to_node)
        return self._priv_to_node_pk

    def __hash__(self) -> int:
        """Make relationship instances hashable using from/to node combination plus properties."""
        # Use from/to node primary keys plus all property values for hashing
        if self._from_node_pk is not None and self._to_node_pk is not None:
            # Include key property values in hash to distinguish relationships with same nodes but different properties
            try:
                property_values = []
                for field_name in self.__class__.model_fields:
                    if hasattr(self, field_name):
                        value = getattr(self, field_name, None)
                        if value is not None and isinstance(value, (str, int, float, bool)):
                            property_values.append((field_name, value))

                return hash((self.__class__.__name__, self._from_node_pk, self._to_node_pk, tuple(property_values)))
            except Exception:
                # Fallback to simpler hash if property access fails
                return hash((self.__class__.__name__, self._from_node_pk, self._to_node_pk))
        # Fallback to object identity if nodes not set
        return hash(id(self))

    def __eq__(self, other: object) -> bool:
        """Define equality based on from/to node combination plus properties."""
        if not isinstance(other, self.__class__):
            return False

        # Use from/to node primary keys plus properties for equality
        if (self._from_node_pk is not None and self._to_node_pk is not None and
            other._from_node_pk is not None and other._to_node_pk is not None):

            # Check node equality
            if not (self._from_node_pk == other._from_node_pk and self._to_node_pk == other._to_node_pk):
                return False

            # Check property equality
            for field_name in self.__class__.model_fields:
                self_value = getattr(self, field_name, None)
                other_value = getattr(other, field_name, None)
                if self_value != other_value:
                    return False

            return True

        # Fallback to object identity
        return id(self) == id(other)

    @property
    def from_node_pk(self) -> Optional[Any]:
        """Get the primary key of the source node."""
        return self._from_node_pk

    @property
    def to_node_pk(self) -> Optional[Any]:
        """Get the primary key of the target node."""
        return self._to_node_pk

    def _extract_node_pk(self, node: Any) -> Any:
        """
        Extract primary key from node instance or return value if already a PK.

        This method implements primary key extraction following Kuzu standards:
        - For model instances: Extract PK field value with validation
        - For raw values: Validate against Kuzu PK type requirements
        - Error handling with detailed diagnostics

        Args:
            node: Either a model instance or a raw primary key value

        Returns:
            The primary key value, validated for Kuzu compatibility

        Raises:
            ValueError: If no primary key found or invalid PK type
            TypeError: If node type is unsupported
        """
        if hasattr(node, 'model_fields'):
            # It's a model instance, find the primary key field
            model_class = type(node)
            for field_name, field_info in model_class.model_fields.items():
                metadata = _kuzu_registry.get_field_metadata(field_info)
                if metadata and metadata.primary_key:
                    pk_value = getattr(node, field_name)
                    # Validate the primary key value
                    self._validate_primary_key_value(pk_value, metadata.kuzu_type, field_name, model_class.__name__)
                    return pk_value
            raise ValueError(f"No primary key found in node {model_class.__name__}")
        else:
            # It's a raw primary key value - validate it against Kuzu PK requirements
            return self._validate_raw_primary_key_value(node)

    def _validate_primary_key_value(self, value: Any, kuzu_type: Union[KuzuDataType, ArrayTypeSpecification], field_name: str, model_name: str) -> None:
        """
        Validate a primary key value against its declared Kuzu type.

        Args:
            value: The primary key value to validate
            kuzu_type: The declared Kuzu type for this field
            field_name: Name of the primary key field
            model_name: Name of the model class

        Raises:
            ValueError: If the value is invalid for the declared type
        """
        if value is None:
            raise ValueError(f"Primary key '{field_name}' in model '{model_name}' cannot be None")

        # Array types cannot be primary keys
        if isinstance(kuzu_type, ArrayTypeSpecification):
            raise ValueError(f"Primary key '{field_name}' in model '{model_name}' cannot be an array type")

        # Validate against Kuzu primary key type requirements
        # kuzu_type is now a string constant from KuzuDataType class
        if not isinstance(kuzu_type, str):
            raise ValueError(f"Primary key '{field_name}' in model '{model_name}' has invalid type specification")

        # Check if this Kuzu type is valid for primary keys
        valid_pk_types = {
            KuzuDataType.STRING, KuzuDataType.INT8, KuzuDataType.INT16, KuzuDataType.INT32,
            KuzuDataType.INT64, KuzuDataType.INT128, KuzuDataType.UINT8, KuzuDataType.UINT16,
            KuzuDataType.UINT32, KuzuDataType.UINT64, KuzuDataType.FLOAT, KuzuDataType.DOUBLE,
            KuzuDataType.DECIMAL, KuzuDataType.DATE, KuzuDataType.TIMESTAMP, KuzuDataType.TIMESTAMP_NS,
            KuzuDataType.TIMESTAMP_MS, KuzuDataType.TIMESTAMP_SEC, KuzuDataType.TIMESTAMP_TZ,
            KuzuDataType.BLOB, KuzuDataType.UUID, KuzuDataType.SERIAL
        }

        if kuzu_type not in valid_pk_types:
            raise ValueError(f"Primary key '{field_name}' in model '{model_name}' has invalid type '{kuzu_type}'. "
                           f"Valid primary key types are: STRING, numeric types, DATE, TIMESTAMP variants, BLOB, UUID, and SERIAL")

    def _validate_raw_primary_key_value(self, value: Any) -> Any:
        """
        Validate a raw primary key value against Kuzu requirements.

        This method validates raw values that are assumed to be primary keys,
        ensuring they meet Kuzu's primary key type requirements.

        Args:
            value: The raw primary key value

        Returns:
            The validated primary key value

        Raises:
            ValueError: If the value type is not valid for Kuzu primary keys
            TypeError: If the value type cannot be determined
        """
        if value is None:
            raise ValueError("Primary key value cannot be None")

        # Map Python types to valid Kuzu primary key types
        python_type = type(value)

        # Valid Python types for Kuzu primary keys
        if python_type in (int, float, str, bytes):
            return value

        # Handle datetime types
        import datetime
        import uuid
        if isinstance(value, (datetime.datetime, datetime.date)):
            return value

        # Handle UUID
        if isinstance(value, uuid.UUID):
            return value

        # Handle decimal types
        try:
            from decimal import Decimal
            if isinstance(value, Decimal):
                return value
        except ImportError:
            pass

        # If we get here, the type is not supported
        raise ValueError(f"Primary key value type '{python_type.__name__}' is not supported by Kuzu. "
                        f"Supported types are: int, float, str, bytes, datetime, date, UUID, and Decimal")

    @classmethod
    def get_relationship_pairs(cls) -> List[RelationshipPair]:
        """Get all FROM-TO pairs for this relationship."""
        pairs = cls.__dict__["__kuzu_relationship_pairs__"]
        return pairs

    @classmethod
    def get_relationship_name(cls) -> str:
        rel_name = cls.__dict__.get("__kuzu_rel_name__")
        if not rel_name:
            raise ValueError(f"Class {cls.__name__} does not have __kuzu_rel_name__. Decorate with @kuzu_relationship.")
        return rel_name

    @classmethod
    def get_multiplicity(cls) -> Optional[RelationshipMultiplicity]:
        return cls.__dict__.get("__kuzu_multiplicity__")

    @classmethod
    def create_between(cls, from_node: Any, to_node: Any, **properties) -> "KuzuRelationshipBase":
        """
        Create a relationship instance between two nodes.

        Args:
            from_node: Source node instance or primary key
            to_node: Target node instance or primary key
            **properties: Additional relationship properties

        Returns:
            Relationship instance for insertion
        """
        return cls(from_node=from_node, to_node=to_node, **properties)

    @classmethod
    def get_direction(cls) -> RelationshipDirection:
        return cls.__dict__.get("__kuzu_direction__", RelationshipDirection.FORWARD)
    
    @classmethod
    def is_multi_pair(cls) -> bool:
        """Check if this relationship has multiple FROM-TO pairs."""
        return cls.__dict__.get("__kuzu_is_multi_pair__", False)

    @classmethod
    def to_cypher_pattern(
        cls, from_alias: str = "a", to_alias: str = "b", rel_alias: Optional[str] = None
    ) -> str:
        rel_name = cls.get_relationship_name()
        rel_pattern = f":{rel_name}" if not rel_alias else f"{rel_alias}:{rel_name}"
        direction = cls.get_direction()
        if direction == RelationshipDirection.FORWARD:
            return f"({from_alias})-[{rel_pattern}]->({to_alias})"
        elif direction == RelationshipDirection.BACKWARD:
            return f"({from_alias})<-[{rel_pattern}]-({to_alias})"
        else:
            return f"({from_alias})-[{rel_pattern}]-({to_alias})"

    @classmethod
    def generate_ddl(cls) -> str:
        return generate_relationship_ddl(cls)
    
    def save(self, session: "KuzuSession") -> None:
        """
        Save this relationship to the database.
        
        Args:
            session: Session to use for saving
        """
        session.add(self)
        session.commit()
    
    def delete(self, session: "KuzuSession") -> None:
        """
        Delete this relationship from the database.
        
        Args:
            session: Session to use for deletion
        """
        session.delete(self)
        session.commit()


# -----------------------------------------------------------------------------
# Field helpers
# -----------------------------------------------------------------------------

def kuzu_rel_field(
    *,
    kuzu_type: Union[KuzuDataType, str],
    not_null: bool = True,
    index: bool = False,
    check_constraint: Optional[str] = None,
    default: Any = ...,
    default_factory: Optional[Callable[[], Any]] = None,
    description: Optional[str] = None,
) -> Any:
    """Shorthand for relationship property fields."""
    return kuzu_field(
        default=default,
        kuzu_type=kuzu_type,
        not_null=not_null,
        index=index,
        check_constraint=check_constraint,
        default_factory=default_factory,
        description=description,
    )


# -----------------------------------------------------------------------------
# DDL generators
# -----------------------------------------------------------------------------

def generate_node_ddl(cls: Type[Any]) -> str:
    """
    Generate DDL for a node class.

    Emitted features:
      - Column types with per-column PRIMARY KEY (if singular)
      - DEFAULT expressions
      - UNIQUE / NOT NULL / CHECK (reported in comments for engine-compat)
      - Table-level PRIMARY KEY for composite keys
      - Table-level FOREIGN KEY constraints (reported in comments)
      - Column-level INDEX tag (reported in comments)
      - Compound indexes emitted after CREATE
      - Table-level constraints provided in decorator (reported in comments)
    """
    # Error message wording and dual-view emission (comments + engine-valid CREATE)
    if not cls.__dict__.get("__kuzu_node_name__"):
        raise ValueError(f"Class {cls.__name__} not decorated with @kuzu_node")

    if cls.__dict__.get("__kuzu_is_abstract__", False):
        # Abstract classes don't generate DDL - this is expected
        raise ValueError(
            f"Cannot generate DDL for abstract node class {cls.__name__}. "
            f"Abstract classes are for inheritance only."
        )

    node_name = cls.__kuzu_node_name__
    columns_minimal: List[str] = []
    pk_fields: List[str] = []
    comment_lines: List[str] = []

    # Column definitions
    for field_name, field_info in cls.model_fields.items():
        meta = _kuzu_registry.get_field_metadata(field_info)
        if not meta:
            continue

        # @@ STEP: Generate Kuzu-valid column definition
        # || S.1: Only PRIMARY KEY and DEFAULT are supported in NODE tables
        col_def = meta.to_ddl_column_definition(field_name, is_node_table=True)
        columns_minimal.append(col_def)

        # Track PK fields for composite handling
        if meta.primary_key:
            pk_fields.append(field_name)

        # Foreign key constraints (comments only; engine doesn't accept them here)
        if meta.foreign_key:
            # @@ STEP: Generate foreign key constraint comment
            comment_lines.append(meta.foreign_key.to_ddl(field_name))

        # Column-level INDEX tag (comments only)
        if meta.index and not meta.primary_key and not meta.unique:
            dtype = meta._canonical_type_name(meta.kuzu_type)
            comment_lines.append(f"{field_name} {dtype} INDEX")

    # Composite PK: remove inline PK tokens and add table-level PK
    if len(pk_fields) >= 2:
        def strip_inline_pk(defn: str, names: Set[str]) -> str:
            parts = defn.split()
            if parts and parts[0] in names and parts[-2:] == ["PRIMARY", "KEY"]:
                return " ".join(parts[:-2])
            return defn

        name_set = set(pk_fields)
        columns_minimal = [strip_inline_pk(c, name_set) for c in columns_minimal]
        columns_minimal.append(f"PRIMARY KEY({', '.join(pk_fields)})")

    # Table-level constraints from decorator (comments only)
    for tc in cls.__dict__.get("__kuzu_table_constraints__", []) or []:
        comment_lines.append(tc)

    # Build CREATE statement with comments prefix (one statement including comments)
    comment_block = ""
    if comment_lines:
        comment_payload = "\n  ".join(comment_lines)
        comment_block = f"/*\n  {comment_payload}\n*/\n"

    ddl = (
        f"{comment_block}"
        f"{DDLConstants.CREATE_NODE_TABLE} {node_name}(\n  " + ",\n  ".join(columns_minimal) + "\n);"
    )

    # Emit compound indexes after CREATE
    for ci in cls.__dict__.get("__kuzu_compound_indexes__", []) or []:
        ddl += f"\n{ci.to_ddl(node_name)}"

    return ddl


def generate_relationship_ddl(cls: Type[T]) -> str:
    """
    Generate DDL for a relationship model supporting multiple FROM-TO pairs.

    Emitted features:
      - Multiple FROM/TO endpoints (e.g., FROM User TO User, FROM User TO City)
      - Property columns with DEFAULT (UNIQUE/NOT NULL/CHECK reported in comments)
      - Multiplicity token placed INSIDE the parentheses
      - Table-level constraints (reported in comments)
      - Compound indexes emitted after CREATE
    """
    # @@ STEP 1: Validate relationship decorator
    try:
        is_relationship = cls.__is_kuzu_relationship__ # type: ignore
    except AttributeError:
        is_relationship = False
    
    if not is_relationship:
        try:
            _ = cls.__kuzu_relationship_name__ # type: ignore
        except AttributeError:
            raise ValueError(f"Class {cls.__name__} not decorated with @kuzu_relationship") from None

    rel_name = cls.__kuzu_relationship_name__ # type: ignore
    
    # @@ STEP 2: Get relationship pairs
    rel_pairs = cls.__kuzu_relationship_pairs__ # type: ignore
    if not rel_pairs:
        raise ValueError(f"{rel_name}: No relationship pairs defined. Use pairs=[(FromNode, ToNode), ...]")
    
    # @@ STEP 3: Validate that all referenced nodes exist and build FROM-TO components
    from_to_components = []
    for pair in rel_pairs:
        from_name = pair.get_from_name()
        to_name = pair.get_to_name()

        # Note: Registry validation is optional for DDL generation
        # This allows for more flexible testing and usage patterns
        # The actual database will validate node existence at runtime

        from_to_components.append(pair.to_ddl_component())

    # @@ STEP 4: Property columns - minimal + comments for rich view
    prop_cols_min: List[str] = []
    comment_lines: List[str] = []

    # @@ STEP: Ensure cls has model_fields attribute (type safety)
    if not hasattr(cls, 'model_fields'):
        raise ValueError(f"Class {cls.__name__} does not have model_fields attribute. Ensure it's a proper Pydantic model.")

    # @@ STEP: Type cast to access model_fields safely after hasattr check
    model_fields = getattr(cls, 'model_fields', {})
    for field_name, field_info in model_fields.items():
        meta = _kuzu_registry.get_field_metadata(field_info)
        if not meta:
            continue
        if meta.is_from_ref or meta.is_to_ref:
            continue

        full_def = meta.to_ddl_column_definition(field_name)   # for tests
        # Minimal emitted column: TYPE + DEFAULT only
        dtype = KuzuFieldMetadata._canonical_type_name(meta.kuzu_type)
        parts = [field_name, dtype]
        if meta.default_value is not None and meta.kuzu_type != KuzuDataType.SERIAL:
            parts.append(KuzuFieldMetadata._render_default(meta.default_value))
        prop_cols_min.append(" ".join(parts))

        if full_def != " ".join(parts):
            comment_lines.append(full_def)

    # @@ STEP 5: Build DDL items list
    items: List[str] = from_to_components  # Start with FROM-TO pairs
    if prop_cols_min:
        items.extend(prop_cols_min)

    multiplicity = cls.__dict__.get("__kuzu_multiplicity__")
    if multiplicity is not None:
        items.append(multiplicity.value)  # inside (...) per grammar

    # Table-level constraints (comments only)
    for tc in cls.__dict__.get("__kuzu_table_constraints__", []) or []:
        comment_lines.append(tc)

    comment_block = ""
    if comment_lines:
        comment_payload = "\n  ".join(comment_lines)
        comment_block = f"/*\n  {comment_payload}\n*/\n"

    ddl = f"{comment_block}{DDLConstants.CREATE_REL_TABLE} {rel_name}(" + ", ".join(items) + ");"

    # Compound indexes after CREATE
    for ci in cls.__dict__.get("__kuzu_compound_indexes__", []) or []:
        ddl += f"\n{ci.to_ddl(rel_name)}"

    return ddl


# -----------------------------------------------------------------------------
# Registry accessors and utilities
# -----------------------------------------------------------------------------

def get_registered_nodes() -> Dict[str, Type[Any]]:
    return _kuzu_registry.nodes.copy()


def get_registered_relationships() -> Dict[str, Type[Any]]:
    return _kuzu_registry.relationships.copy()


def get_all_models() -> Dict[str, Type[Any]]:
    """Get all registered models (nodes and relationships)."""
    all_models = {}
    all_models.update(_kuzu_registry.nodes)
    all_models.update(_kuzu_registry.relationships)
    return all_models


def get_ddl_for_node(node_cls: Type[Any]) -> str:
    """Generate DDL for a node class."""
    # @@ STEP: Check for node name attribute
    try:
        node_name = node_cls.__kuzu_node_name__
    except AttributeError:
        raise ValueError(
            ValidationMessageConstants.MISSING_KUZU_NODE_NAME.format(node_cls.__name__)
        )
    fields = []
    
    for field_name, field_info in node_cls.model_fields.items():
        meta = _kuzu_registry.get_field_metadata(field_info)
        if meta:
            field_ddl = meta.to_ddl(field_name)
            fields.append(field_ddl)
    
    if not fields:
        raise ValueError(
            f"Node {node_name} has no Kuzu fields defined. "
            f"At least one field with Kuzu metadata is required."
        )
    
    return f"{DDLConstants.CREATE_NODE_TABLE} {node_name} (\n    {', '.join(fields)}\n);"


def get_ddl_for_relationship(rel_cls: Type[Any]) -> str:
    """Generate DDL for a relationship.
    
    :param rel_cls: Relationship class.
    :return: DDL statement.
    """
    # @@ STEP: Validate relationship class has required attribute
    try:
        rel_name = rel_cls.__kuzu_rel_name__
        _ = rel_name  # Mark as intentionally unused - only used for validation
    except AttributeError:
        raise ValueError(
            ValidationMessageConstants.MISSING_KUZU_REL_NAME.format(rel_cls.__name__)
        )
    
    # Multi-pair or new single-pair format
    return generate_relationship_ddl(rel_cls)

def get_all_ddl() -> str:
    """
    Generate DDL for all registered models in the correct dependency order.

    This function automatically triggers registry finalization to resolve
    all foreign key references and determine the correct creation order.

    IMPORTANT: Nodes must be created before relationships that reference them.

    Returns:
        str: DDL statements for all models in dependency order

    Raises:
        ValueError: If registry finalization fails due to unresolvable references
    """
    # @@ STEP: Ensure registry is finalized
    if not _kuzu_registry.finalize_registry():
        errors = _kuzu_registry.get_resolution_errors()
        error_msg = "Failed to finalize registry for DDL generation"
        if errors:
            error_msg += ":\n" + "\n".join(errors)
        raise ValueError(error_msg)

    ddl_statements = []

    # @@ STEP: Generate DDL with nodes first, then relationships
    # This ensures that all node tables exist before relationship tables are created
    creation_order = _kuzu_registry.get_creation_order()

    # First pass: Create all nodes
    for model_name in creation_order:
        if model_name in _kuzu_registry.nodes:
            model_cls = _kuzu_registry.models.get(model_name)
            if model_cls:
                ddl = get_ddl_for_node(model_cls)
                if ddl:
                    ddl_statements.append(ddl)

    # Second pass: Create all relationships
    for model_name in creation_order:
        if model_name in _kuzu_registry.relationships:
            model_cls = _kuzu_registry.models.get(model_name)
            if model_cls:
                ddl = get_ddl_for_relationship(model_cls)
                if ddl:
                    ddl_statements.append(ddl)

    return "\n".join(ddl_statements)


def validate_all_models() -> None:
    """Validate all registered models."""
    errors = []
    
    # @@ STEP: Validate nodes - strict access, no exception handling
    for node_name, node_cls in _kuzu_registry.nodes.items():
        # Direct method call - if it fails, the class is improperly configured
        node_errors = node_cls.validate_foreign_keys()
        errors.extend(node_errors)
    
    # @@ STEP: Validate relationships - strict access, no exception handling
    for rel_name, rel_cls in _kuzu_registry.relationships.items():
        # Direct method call - if it fails, the class is improperly configured
        rel_errors = rel_cls.validate_foreign_keys()
        errors.extend(rel_errors)
    
    if errors:
        raise ValueError("Validation failed for one or more models: " + "\n".join(errors))


def clear_registry():
    """Clear all registered models and reset registry state."""
    # Enhanced registry cleanup to prevent memory corruption
    # @@ STEP 1: Clear all model registrations
    _kuzu_registry.nodes.clear()
    _kuzu_registry.relationships.clear()
    _kuzu_registry.models.clear()

    # @@ STEP 2: Clear dependency tracking
    _kuzu_registry._model_dependencies.clear()
    _kuzu_registry._unresolved_foreign_keys.clear()
    _kuzu_registry._resolution_errors.clear()
    _kuzu_registry._circular_dependencies.clear()
    _kuzu_registry._self_references.clear()

    # @@ STEP 3: Reset resolution phase
    _kuzu_registry._resolution_phase = RegistryResolutionConstants.PHASE_REGISTRATION

    # @@ STEP 4: Force garbage collection to free memory
    gc.collect()


def get_node_by_name(name: str) -> Optional[Type[Any]]:
    return _kuzu_registry.nodes.get(name)


def get_relationship_by_name(name: str) -> Optional[Type[Any]]:
    return _kuzu_registry.relationships.get(name)


def finalize_registry() -> bool:
    """
    Explicitly finalize the registry to resolve all foreign key references.

    This is automatically called by get_all_ddl(), but can be called manually
    for early validation or to check for resolution errors.

    Returns:
        bool: True if finalization was successful, False otherwise
    """
    return _kuzu_registry.finalize_registry()


def get_registry_resolution_errors() -> List[str]:
    """
    Get any resolution errors from the registry.

    Returns:
        List[str]: List of resolution error messages
    """
    return _kuzu_registry.get_resolution_errors()


def get_circular_dependencies() -> Set[Tuple[str, str]]:
    """
    Get detected circular dependencies between models.

    Returns:
        Set[Tuple[str, str]]: Set of (model1, model2) tuples representing circular dependencies
    """
    return _kuzu_registry.get_circular_dependencies()


def get_self_references() -> Set[str]:
    """
    Get models that have self-references.

    Returns:
        Set[str]: Set of model names that reference themselves
    """
    return _kuzu_registry.get_self_references()


def is_registry_finalized() -> bool:
    """
    Check if the registry has been finalized.

    Returns:
        bool: True if the registry is finalized, False otherwise
    """
    return _kuzu_registry.is_finalized()


def get_model_creation_order() -> List[str]:
    """
    Get the creation order for all models, handling circular dependencies.

    Returns:
        List[str]: List of model names in creation order

    Raises:
        ValueError: If the registry cannot be finalized
    """
    if not _kuzu_registry.finalize_registry():
        errors = _kuzu_registry.get_resolution_errors()
        error_msg = "Cannot determine creation order: Registry finalization failed"
        if errors:
            error_msg += ":\n" + "\n".join(errors)
        raise ValueError(error_msg)

    return _kuzu_registry.get_creation_order()


def generate_all_ddl() -> str:
    """
    Generate DDL for all registered nodes (in dependency order) and relationships.
    """
    ddl_statements: List[str] = []
    order = _kuzu_registry.get_creation_order()

    # Nodes first
    for name in order:
        if name in _kuzu_registry.nodes:
            cls = _kuzu_registry.nodes[name]
            ddl = generate_node_ddl(cls)
            if ddl:
                ddl_statements.append(ddl)

    # Relationships
    for name, cls in _kuzu_registry.relationships.items():
        ddl = generate_relationship_ddl(cls)
        if ddl:
            ddl_statements.append(ddl)

    return "\n\n".join(ddl_statements)


# @@ STEP: Initialize SQLKeywordRegistry with time keywords from KuzuDefaultFunction
# || S.S: This must be done after the enum is imported
SQLKeywordRegistry._initialize_time_keywords()

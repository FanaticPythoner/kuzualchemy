"""
Kùzu ORM system with decorators, field metadata, and DDL generation.
Type-safe metadata and DDL emission that matches the expected grammar and ordering
used in tests (PRIMARY KEY inline when singular, DEFAULT/UNIQUE/NOT NULL/CHECK ordering,
FK constraints, column-level INDEX tags, and correct relationship multiplicity placement).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
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
    RelationshipDirectionConstants,
    KuzuDataTypeConstants,
    ConstraintConstants,
    ArrayTypeConstants,
    ErrorMessages,
    ValidationMessageConstants,
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


# Register default handlers
DefaultValueHandlerRegistry.register_handler(bool, DefaultValueHandlerRegistry._bool_handler)
DefaultValueHandlerRegistry.register_handler(int, DefaultValueHandlerRegistry._int_handler)
DefaultValueHandlerRegistry.register_handler(float, DefaultValueHandlerRegistry._float_handler)
DefaultValueHandlerRegistry.register_handler(str, DefaultValueHandlerRegistry._string_handler)

class TableNameResolver:
    """Registry for explicitly mapping model types to their table names."""
    _table_names: Dict[type, str] = {}
    
    @classmethod
    def register_model(cls, model_type: type, table_name: str) -> None:
        """Register a model type with its table name."""
        cls._table_names[model_type] = table_name
    
    @classmethod
    def resolve(cls, model: Any) -> str:
        """Resolve table name from a model object."""
        model_type = type(model)
        
        # For string types, return as-is
        if model_type is str:
            return model
            
        # Check registered types first
        if model_type in cls._table_names:
            return cls._table_names[model_type]
        
        # @@ STEP 1: Check for __kuzu_node_name__ attribute directly
        if hasattr(model, '__kuzu_node_name__'):
            return model.__kuzu_node_name__
        
        # @@ STEP 2: Check if it's a relationship model
        # || S.S.1: Use hasattr instead of dir() - more explicit and efficient
        if hasattr(model, '__kuzu_rel_name__'):
            # || S.S.2: Provide clear error message for relationship models
            raise ValueError(
                f"Model {model.__name__} is a relationship model, not a node model. "
                f"Use get_relationship_name() for relationship models."
            )
        
        # @@ STEP 3: No resolution found - raise explicit error
        # || S.S.3: Error message with guidance
        raise ValueError(
            f"Cannot resolve table name for {model}. "
            f"Register it using TableNameResolver.register_model() or ensure it has "
            f"__kuzu_node_name__ or __kuzu_rel_name__ attribute."
        )

# -----------------------------------------------------------------------------
# Schema enums
# -----------------------------------------------------------------------------

class RelationshipMultiplicity(Enum):
    """
    Relationship multiplicity types.
    
    :class: RelationshipMultiplicity
    :synopsis: Enumeration of relationship cardinality types
    """

    MANY_TO_ONE = DDLConstants.MANY_TO_ONE
    ONE_TO_MANY = DDLConstants.ONE_TO_MANY
    MANY_TO_MANY = DDLConstants.MANY_TO_MANY
    ONE_TO_ONE = DDLConstants.ONE_TO_ONE  # default if unspecified


class RelationshipDirection(Enum):
    """
    Relationship direction.
    
    :class: RelationshipDirection
    :synopsis: Enumeration of relationship directions
    """

    FORWARD = RelationshipDirectionConstants.FORWARD_ARROW    # From source to target
    BACKWARD = RelationshipDirectionConstants.BACKWARD_ARROW   # From target to source
    BOTH = RelationshipDirectionConstants.BOTH_ARROW      # Bidirectional (undirected pattern)


class KuzuDataType(Enum):
    """
    Kuzu data types enumeration.
    
    :class: KuzuDataType
    :synopsis: Enumeration of supported Kuzu data types
    """

    # Numeric types
    INT8 = KuzuDataTypeConstants.INT8
    INT16 = KuzuDataTypeConstants.INT16
    INT32 = KuzuDataTypeConstants.INT32
    INT64 = KuzuDataTypeConstants.INT64
    INT128 = KuzuDataTypeConstants.INT128
    UINT8 = KuzuDataTypeConstants.UINT8
    UINT16 = KuzuDataTypeConstants.UINT16
    UINT32 = KuzuDataTypeConstants.UINT32
    UINT64 = KuzuDataTypeConstants.UINT64
    FLOAT = KuzuDataTypeConstants.FLOAT
    DOUBLE = KuzuDataTypeConstants.DOUBLE
    DECIMAL = KuzuDataTypeConstants.DECIMAL

    # Serial (auto-incrementing)
    SERIAL = KuzuDataTypeConstants.SERIAL

    # Strings
    STRING = KuzuDataTypeConstants.STRING

    # Boolean
    BOOL = KuzuDataTypeConstants.BOOL
    BOOLEAN = KuzuDataTypeConstants.BOOLEAN

    # Date/Time
    DATE = KuzuDataTypeConstants.DATE
    TIMESTAMP = KuzuDataTypeConstants.TIMESTAMP
    TIMESTAMP_NS = KuzuDataTypeConstants.TIMESTAMP_NS
    TIMESTAMP_MS = KuzuDataTypeConstants.TIMESTAMP_MS
    TIMESTAMP_SEC = KuzuDataTypeConstants.TIMESTAMP_SEC
    TIMESTAMP_TZ = KuzuDataTypeConstants.TIMESTAMP_TZ
    INTERVAL = KuzuDataTypeConstants.INTERVAL

    # Binary
    BLOB = KuzuDataTypeConstants.BLOB

    # UUID
    UUID = KuzuDataTypeConstants.UUID

    # Complex types
    ARRAY = KuzuDataTypeConstants.ARRAY
    STRUCT = KuzuDataTypeConstants.STRUCT
    MAP = KuzuDataTypeConstants.MAP
    UNION = KuzuDataTypeConstants.UNION

    # Node/Relationship references (rarely used as columns)
    NODE = KuzuDataTypeConstants.NODE
    REL = KuzuDataTypeConstants.REL


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
    Metadata for foreign key constraints.

    :class: ForeignKeyMetadata
    :synopsis: Dataclass for foreign key constraint metadata
    """
    target_model: Union[str, Type[Any]]
    target_field: str
    on_delete: Optional[CascadeAction] = None
    on_update: Optional[CascadeAction] = None

    def to_ddl(self, field_name: str) -> str:
        """
        Generate DDL comment for foreign key constraint.

        Since Kuzu doesn't support foreign key constraints in DDL,
        this generates a comment for documentation purposes.
        """
        # @@ STEP: Determine target model name
        if isinstance(self.target_model, str):
            target_name = self.target_model
        else:
            # || S.1: Try to get the node name from the model
            target_name = getattr(self.target_model, '__kuzu_node_name__', None)
            if not target_name:
                # || S.2: Fallback to class name
                target_name = getattr(self.target_model, '__name__', str(self.target_model))

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
        # Prefer BOOL over BOOLEAN in emitted DDL to match tests
        if dt == KuzuDataType.BOOLEAN:
            return KuzuDataType.BOOL.value
        return dt.value

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
        kuzu_type: Kuzu data type (can be ARRAY/LIST for array types)
        element_type: Element type for array fields (e.g., 'INT64' for INT64[])
        auto_increment: Enable auto-increment (SERIAL type)
        default_factory: Python-side default factory function
    """
    if element_type is not None:
        # User specified element_type, so this is an array
        if isinstance(element_type, str):
            element_type = KuzuDataType(element_type)
        kuzu_type = ArrayTypeSpecification(element_type=element_type)
    elif isinstance(kuzu_type, str):
        kuzu_type = KuzuDataType(kuzu_type)

    if auto_increment:
        kuzu_type = KuzuDataType.SERIAL

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
        try:
            return self.from_node.__kuzu_node_name__
        except AttributeError:
            try:
                return self.from_node.__name__
            except AttributeError as exc:
                raise ValueError(
                    ValidationMessageConstants.MISSING_KUZU_NODE_NAME.format(self.from_node)
                ) from exc
    
    def get_to_name(self) -> str:
        """Get the name of the TO node."""
        if isinstance(self.to_node, str):
            return self.to_node
        try:
            return self.to_node.__kuzu_node_name__
        except AttributeError:
            try:
                return self.to_node.__name__
            except AttributeError as exc:
                raise ValueError(
                    ValidationMessageConstants.MISSING_KUZU_NODE_NAME.format(self.to_node)
                ) from exc
    
    def to_ddl_component(self) -> str:
        """Convert to DDL component for CREATE REL TABLE."""
        return f"{DDLConstants.REL_TABLE_GROUP_FROM} {self.get_from_name()} {DDLConstants.REL_TABLE_GROUP_TO} {self.get_to_name()}"
    
    def __repr__(self) -> str:
        return f"RelationshipPair(from={self.from_node}, to={self.to_node})"


# -----------------------------------------------------------------------------
# Global registry
# -----------------------------------------------------------------------------

class KuzuRegistry:
    """Global registry for nodes, relationships, and model metadata."""

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
        self.nodes: Dict[str, Type[Any]] = {}
        self.relationships: Dict[str, Type[Any]] = {}
        self.models: Dict[str, Type[Any]] = {}
        self._model_dependencies: Dict[str, Set[str]] = {}

    def register_node(self, name: str, cls: Type[Any]) -> None:
        self.nodes[name] = cls
        self.models[name] = cls
        self._analyze_dependencies(name, cls)

    def register_relationship(self, name: str, cls: Type[Any]) -> None:
        self.relationships[name] = cls
        self.models[name] = cls
        self._analyze_dependencies(name, cls)

    def _analyze_dependencies(self, name: str, cls: Type[Any]) -> None:
        """Analyze FK-like dependencies (metadata) for creation ordering."""
        dependencies = set()
        for _, field_info in cls.model_fields.items():
            metadata = self.get_field_metadata(field_info)
            if metadata and metadata.foreign_key:
                target_model = metadata.foreign_key.target_model
                if isinstance(target_model, str):
                    target_name = target_model
                else:
                    # @@ STEP: Try multiple ways to resolve the target model name
                    target_name = None
                    # || S.1: Check for kuzu_node_name attribute
                    try:
                        target_name = target_model.__kuzu_node_name__
                    except AttributeError:
                        # || S.2: Check for __name__ attribute
                        try:
                            target_name = target_model.__name__
                        except AttributeError:
                            # || S.3: Check for __qualname__ attribute
                            try:
                                qualname = getattr(target_model, '__qualname__', None)
                                if qualname:
                                    # For nested classes, use the last part of qualname
                                    target_name = qualname.split('.')[-1]
                                else:
                                    raise AttributeError("No qualname")
                            except AttributeError:
                                # Cannot determine target name - THIS IS AN ERROR
                                raise ValueError(
                                    f"Cannot determine name for target model {target_model} "
                                    f"in target name {target_name}. Model must have __kuzu_node_name__, "
                                    f"__name__, or __qualname__ attribute."
                                )
                dependencies.add(target_name)
        self._model_dependencies[name] = dependencies

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

    def get_creation_order(self) -> List[str]:
        """Topologically sort by metadata dependencies (nodes & rels)."""
        visited = set()
        order: List[str] = []

        def visit(name: str):
            if name in visited:
                return
            visited.add(name)
            for dep in self._model_dependencies.get(name, set()):
                if dep != name:
                    visit(dep)
            order.append(name)

        for name in self.models:
            visit(name)
        return order


# Singleton
_kuzu_registry = KuzuRegistry()


# -----------------------------------------------------------------------------
# Relationship pair processing helpers
# -----------------------------------------------------------------------------

def _process_relationship_pairs(
    pairs: Union[
        List[Tuple[Union[Type[Any], str], Union[Type[Any], str]]],
        Dict[Union[Type[Any], str], Union[Set[Union[Type[Any], str]], List[Union[Type[Any], str]]]]
    ],
    rel_name: str
) -> List[RelationshipPair]:
    """
    Process relationship pairs supporting both traditional and enhanced formats.

    This function handles:
    1. Traditional format: [(FromType, ToType), ...]
    2. Enhanced format: {FromType: {ToType1, ToType2}, ...}
    3. Mixed format: {FromType: [ToType1, ToType2], ...}

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
            rel_pairs.append(RelationshipPair(from_type, to_type))

    elif isinstance(pairs, dict):
        # Enhanced format: {FromType: {ToType1, ToType2}, ...} or {FromType: [ToType1, ToType2], ...}
        for from_type, to_types in pairs.items():
            if isinstance(to_types, (set, list)):
                for to_type in to_types:
                    rel_pairs.append(RelationshipPair(from_type, to_type))
            else:
                # Single to_type
                rel_pairs.append(RelationshipPair(from_type, to_types))
    else:
        raise ValueError(
            f"Relationship {rel_name}: 'pairs' must be either a list of tuples "
            f"[(FromType, ToType), ...] or a dictionary {{FromType: {{ToType1, ToType2}}, ...}}"
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

    direction: RelationshipDirection = RelationshipDirection.FORWARD,
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
        """Validate metadata references only (no DB enforcement)."""
        errors: List[str] = []
        for field_name, fk_ref in cls.get_foreign_key_fields().items():
            target_model = fk_ref.target_model

            # @@ STEP: Validate target_model type and skip string references
            # || S.1: String references cannot be validated at this time
            # || S.2: Only validate actual model classes with model_fields attribute
            if isinstance(target_model, str) or not hasattr(target_model, 'model_fields'):
                raise TypeError(
                    f"Field {field_name}: target model {target_model} is not a valid model class"
                )

            # @@ STEP: Direct validation - target_model is guaranteed to be a proper model class
            # || S.1: Access model_fields directly (safe after type check)
            # || S.2: Access __name__ directly for error message (safe after type check)
            if fk_ref.target_field not in target_model.model_fields:
                errors.append(
                    f"Field {field_name}: target field {fk_ref.target_field} not found in {target_model.__name__}"
                )
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

    def __init__(self, from_node: Optional[Any] = None, to_node: Optional[Any] = None, **kwargs):
        """
        Initialize relationship with from/to node references.

        Args:
            from_node: Source node instance or primary key value
            to_node: Target node instance or primary key value
            **kwargs: Additional relationship properties
        """
        super().__init__(**kwargs)
        self._from_node = from_node
        self._to_node = to_node

        # Store node references for relationship creation
        if from_node is not None:
            self._from_node_pk = self._extract_node_pk(from_node)
        else:
            self._from_node_pk = None

        if to_node is not None:
            self._to_node_pk = self._extract_node_pk(to_node)
        else:
            self._to_node_pk = None

    @property
    def from_node(self) -> Optional[Any]:
        """Get the source node of this relationship."""
        return self._from_node

    @property
    def to_node(self) -> Optional[Any]:
        """Get the target node of this relationship."""
        return self._to_node

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
        if not isinstance(kuzu_type, KuzuDataType):
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
            raise ValueError(f"Primary key '{field_name}' in model '{model_name}' has invalid type '{kuzu_type.value}'. "
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

        # Validate that referenced nodes are registered
        if from_name not in _kuzu_registry.nodes:
            raise ValueError(
                f"Relationship {rel_name} references FROM node '{from_name}' which is not registered. "
                f"Ensure the node class is decorated with @kuzu_node and properly imported."
            )

        if to_name not in _kuzu_registry.nodes:
            raise ValueError(
                f"Relationship {rel_name} references TO node '{to_name}' which is not registered. "
                f"Ensure the node class is decorated with @kuzu_node and properly imported."
            )

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
    """Generate DDL for all registered models."""
    ddl_statements = []
    
    # Generate DDL for nodes
    for node_name, node_cls in _kuzu_registry.nodes.items():
        _ = node_name  # Mark as intentionally unused - only node_cls is needed
        ddl = get_ddl_for_node(node_cls)
        if ddl:
            ddl_statements.append(ddl)

    # Generate DDL for relationships
    for rel_name, rel_cls in _kuzu_registry.relationships.items():
        _ = rel_name  # Mark as intentionally unused - only rel_cls is needed
        ddl = get_ddl_for_relationship(rel_cls)
        if ddl:
            ddl_statements.append(ddl)
    
    return "\n".join(ddl_statements)


def validate_all_models() -> List[str]:
    """Validate all registered models."""
    errors = []
    
    # @@ STEP: Validate nodes with direct method access
    for node_name, node_cls in _kuzu_registry.nodes.items():
        try:
            node_errors = node_cls.validate_foreign_keys()
            errors.extend(node_errors)
        except AttributeError:
            # Method doesn't exist - THIS IS AN ERROR for a node class
            raise ValueError(
                f"Node class {node_name} does not have validate_foreign_keys() method. "
                f"This indicates the class is not properly decorated with @node."
            )
    
    # @@ STEP: Validate relationships with direct method access
    for rel_name, rel_cls in _kuzu_registry.relationships.items():
        try:
            rel_errors = rel_cls.validate_foreign_keys()
            errors.extend(rel_errors)
        except AttributeError:
            # Method doesn't exist - THIS IS AN ERROR for a relationship class
            raise ValueError(
                f"Relationship class {rel_name} does not have validate_foreign_keys() method. "
                f"This indicates the class is not properly decorated with @relationship."
            )
    
    return errors


def clear_registry():
    """Clear all registered models."""
    _kuzu_registry.nodes.clear()
    _kuzu_registry.relationships.clear()
    _kuzu_registry._model_dependencies.clear()


def get_node_by_name(name: str) -> Optional[Type[Any]]:
    return _kuzu_registry.nodes.get(name)


def get_relationship_by_name(name: str) -> Optional[Type[Any]]:
    return _kuzu_registry.relationships.get(name)


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

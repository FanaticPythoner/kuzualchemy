"""
KuzuAlchemy - A SQLAlchemy-like ORM for Kuzu graph database.

This package provides a comprehensive ORM for Kuzu graph database with:
- SQLAlchemy-like query interface
- Pydantic-based model definitions
- Relationship management
- Session and transaction support
- Type-safe field definitions

Note: This software is currently in alpha development. APIs may change.
"""

from __future__ import annotations

# Core ORM components
from .kuzu_orm import (
    # Base models
    KuzuBaseModel,
    KuzuRelationshipBase,

    # Decorators
    kuzu_node,
    kuzu_relationship,
    
    # Field functions
    kuzu_field,
    foreign_key,
    
    # Metadata classes
    KuzuFieldMetadata,
    ForeignKeyMetadata,
    CheckConstraintMetadata,
    IndexMetadata,
    ForeignKeyReference,
    CompoundIndex,
    
    # Enums
    KuzuDataType,
    RelationshipMultiplicity,
    RelationshipDirection,
    
    # Registry and utilities
    get_registered_nodes,
    get_registered_relationships,
    get_all_models,
    get_all_ddl,
    get_ddl_for_node,
    get_ddl_for_relationship,
    validate_all_models,
    clear_registry,
    KuzuRegistry,

    # Enhanced registry functions
    finalize_registry,
    get_registry_resolution_errors,
    get_circular_dependencies,
    get_self_references,
    is_registry_finalized,
    get_model_creation_order,
)


# Constants
from .constants import (
    DatabaseConstants,
    DDLConstants,
    CypherConstants,
    ModelMetadataConstants,
    SessionConstants,
    QueryConstants,
    ErrorMessages,
    ValidationConstants,
    LoggingConstants,
    TypeMappingConstants,
    PerformanceConstants,
    FileSystemConstants,
    PackageConstants,
    CascadeAction,
)

# Query system
from .kuzu_query import Query
from .kuzu_query_builder import CypherQueryBuilder
from .kuzu_query_fields import ModelFieldAccessor, QueryField
from .kuzu_query_expressions import (
    FilterExpression,
    FieldFilterExpression,
    CompoundFilterExpression,
    NotFilterExpression,
    RawCypherExpression,
    BetweenExpression,
    ArithmeticExpression,
    FunctionExpression,
    TemporalExpression,
    PatternExpression,
    ComparisonOperator,
    LogicalOperator,
    AggregateFunction,
    OrderDirection,
    JoinType,
    ArithmeticOperator,
    StringOperator,
    PatternOperator,
    TemporalOperator,
)

# Session management
from .kuzu_session import (
    KuzuSession,
    SessionFactory,
    KuzuConnection,
    KuzuTransaction,
)

# Create session_scope alias
session_scope = SessionFactory.session_scope

# Base model with enum conversion
from .BaseModel import BaseModel

# Kuzu functions
from .kuzu_functions import (
    # Text functions
    concat, ws_concat, array_extract, array_slice, list_element, list_extract, prefix, suffix,
    contains, ends_with, lower, lcase, left, levenshtein, lpad, ltrim, repeat, reverse, right,
    rpad, rtrim, starts_with, substring, substr, trim, upper, ucase, initcap, string_split, split_part,

    # Pattern matching functions
    regexp_matches, regexp_replace, regexp_extract, regexp_extract_all, regexp_split_to_array,

    # Numeric functions
    pi, abs, ceil, ceiling, floor, round, sqrt, pow, sin, cos, tan, asin, acos, atan, atan2,
    ln, log, log2, log10, negate, sign, even, factorial, gamma, lgamma, bitwise_xor, cot, degrees, radians,

    # List functions
    list_creation, size, list_concat, range, list_transform, list_filter, list_reduce, list_has_all,
    all_func, any_func, none_func, single_func,
    # Additional list functions
    list_cat, array_concat, array_cat, list_append, array_append, array_push_back,
    list_prepend, array_prepend, array_push_front, list_position, list_indexof, array_position, array_indexof,
    list_contains, list_has, array_contains, array_has, list_slice, list_reverse, list_sort, list_reverse_sort,
    list_sum, list_product, list_distinct, list_unique, list_any_value, list_to_string,

    # Utility functions
    coalesce, ifnull, nullif, typeof, constant_or_null, count_if, error,

    # Date/time functions
    current_date, current_timestamp, date_part, date_trunc, datepart, datetrunc,
    dayname, monthname, last_day, greatest, least, make_date, century, epoch_ms, to_epoch_ms,

    # Hash functions
    md5, sha256, hash,

    # UUID functions
    gen_random_uuid, uuid,

    # Casting functions
    cast, to_int64, to_int32, to_int16, to_double, to_float, to_string, to_date, to_timestamp,
    cast_as, case,

    # Interval functions
    to_years, to_months, to_days, to_hours, to_minutes, to_seconds, to_milliseconds, to_microseconds,

    # Array functions
    array_value, array_distance, array_squared_distance, array_dot_product, array_inner_product,
    array_cross_product, array_cosine_similarity,

    # Blob functions
    blob, encode, decode, octet_length,

    # Struct functions
    struct_extract,

    # Map functions
    map_func, map_extract, element_at, cardinality, map_keys, map_values,

    # Union functions
    union_value, union_tag, union_extract,

    # Node/rel functions
    id_func, label, labels, offset,

    # Recursive rel functions
    nodes, rels, properties, is_trail, is_acyclic, length, cost,
)

# Version info
__version__ = PackageConstants.VERSION
__author__ = PackageConstants.AUTHOR
__email__ = PackageConstants.EMAIL
__license__ = PackageConstants.LICENSE

# Public API
__all__ = [
    # Base model
    "KuzuBaseModel",
    "BaseModel",
    # Main Decorators
    "kuzu_node",
    "kuzu_relationship",
    "kuzu_field",
    # Enums and metadata classes
    "KuzuDataType",
    "RelationshipDirection",
    "RelationshipMultiplicity",
    "CascadeAction",
    "ForeignKeyReference",
    "CompoundIndex",
    "ForeignKeyMetadata",
    "CheckConstraintMetadata",
    "IndexMetadata",
    "KuzuFieldMetadata",
    # Query classes
    "Query",
    "CypherQueryBuilder",
    "ModelFieldAccessor",
    "QueryField",
    "FilterExpression",
    "FieldFilterExpression",
    "CompoundFilterExpression",
    "NotFilterExpression",
    "RawCypherExpression",
    "BetweenExpression",
    "ArithmeticExpression",
    "FunctionExpression",
    "TemporalExpression",
    "PatternExpression",
    "ComparisonOperator",
    "LogicalOperator",
    "AggregateFunction",
    "OrderDirection",
    "JoinType",
    "ArithmeticOperator",
    "StringOperator",
    "PatternOperator",
    "TemporalOperator",
    # Session classes
    "KuzuSession",
    "SessionFactory",
    "session_scope",
    # Registry functions
    "get_registered_nodes",
    "get_registered_relationships",
    "get_all_models",
    "get_all_ddl",
    "get_ddl_for_node",
    "get_ddl_for_relationship",
    "validate_all_models",
    "clear_registry",
    "KuzuRegistry",
    # Enhanced registry functions
    "finalize_registry",
    "get_registry_resolution_errors",
    "get_circular_dependencies",
    "get_self_references",
    "is_registry_finalized",
    "get_model_creation_order",
    # Constants
    "DatabaseConstants",
    "DDLConstants",
    "CypherConstants",
    "ModelMetadataConstants",
    "SessionConstants",
    "QueryConstants",
    "ErrorMessages",
    "ValidationConstants",
    "LoggingConstants",
    "TypeMappingConstants",
    "PerformanceConstants",
    "FileSystemConstants",
    "PackageConstants",
    # Kuzu functions
    # Text functions
    "concat", "ws_concat", "array_extract", "array_slice", "list_element", "list_extract", "prefix", "suffix",
    "contains", "ends_with", "lower", "lcase", "left", "levenshtein", "lpad", "ltrim", "repeat", "reverse", "right",
    "rpad", "rtrim", "starts_with", "substring", "substr", "trim", "upper", "ucase", "initcap", "string_split", "split_part",
    # Pattern matching functions
    "regexp_matches", "regexp_replace", "regexp_extract", "regexp_extract_all", "regexp_split_to_array",
    # Numeric functions
    "pi", "abs", "ceil", "ceiling", "floor", "round", "sqrt", "pow", "sin", "cos", "tan",
    "asin", "acos", "atan", "atan2", "ln", "log", "log2", "log10", "negate", "sign", "even",
    "factorial", "gamma", "lgamma", "bitwise_xor", "cot", "degrees", "radians",
    # List functions
    "list_creation", "size", "list_concat", "range", "list_transform", "list_filter", "list_reduce",
    "list_has_all", "all_func", "any_func", "none_func", "single_func",
    "list_cat", "array_concat", "array_cat", "list_append", "array_append", "array_push_back",
    "list_prepend", "array_prepend", "array_push_front", "list_position", "list_indexof", "array_position", "array_indexof",
    "list_contains", "list_has", "array_contains", "array_has", "list_slice", "list_reverse", "list_sort", "list_reverse_sort",
    "list_sum", "list_product", "list_distinct", "list_unique", "list_any_value", "list_to_string",
    # Utility functions
    "coalesce", "ifnull", "nullif", "typeof", "constant_or_null", "count_if", "error",
    # Date/time functions
    "current_date", "current_timestamp", "date_part", "date_trunc", "datepart", "datetrunc",
    "dayname", "monthname", "last_day", "greatest", "least", "make_date", "century", "epoch_ms", "to_epoch_ms",
    # Hash functions
    "md5", "sha256", "hash",
    # UUID functions
    "gen_random_uuid", "uuid",
    # Casting functions
    "cast", "to_int64", "to_int32", "to_int16", "to_double", "to_float", "to_string", "to_date", "to_timestamp",
    "cast_as", "case",
    # Interval functions
    "to_years", "to_months", "to_days", "to_hours", "to_minutes", "to_seconds", "to_milliseconds", "to_microseconds",
    # Array functions
    "array_value", "array_distance", "array_squared_distance", "array_dot_product", "array_inner_product",
    "array_cross_product", "array_cosine_similarity",
    # Blob functions
    "blob", "encode", "decode", "octet_length",
    # Struct functions
    "struct_extract",
    # Map functions
    "map_func", "map_extract", "element_at", "cardinality", "map_keys", "map_values",
    # Union functions
    "union_value", "union_tag", "union_extract",
    # Node/rel functions
    "id_func", "label", "labels", "offset",
    # Recursive rel functions
    "nodes", "rels", "properties", "is_trail", "is_acyclic", "length", "cost",
]

# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
Constants module for KuzuAlchemy ORM.

This module centralizes all constants, configuration values, and literal strings
used throughout the KuzuAlchemy codebase. No magic values are allowed elsewhere.

:module: constants
:synopsis: Centralized constants and configuration for KuzuAlchemy ORM
:author: KuzuAlchemy Contributors
"""

from __future__ import annotations

from enum import Enum, StrEnum
from typing import Final, Any

from .kuzu_function_types import TimeFunction, UUIDFunction, SequenceFunction


# ============================================================================
# KUZU DEFAULT VALUE FUNCTIONS
# ============================================================================

class KuzuDefaultFunction(Enum):
    """Master enum with properly typed default value functions."""

    # @@ STEP 1: Time functions - using TimeFunction class
    CURRENT_TIMESTAMP = TimeFunction("current_timestamp()")
    CURRENT_DATE = TimeFunction("current_date()")
    CURRENT_TIME = TimeFunction("current_time()")
    NOW = TimeFunction("now()")

    # @@ STEP 2: UUID functions - using UUIDFunction class
    GEN_RANDOM_UUID = UUIDFunction("gen_random_uuid()")
    UUID = UUIDFunction("uuid()")

    # @@ STEP 3: Sequence functions - using SequenceFunction class
    NEXTVAL = SequenceFunction("nextval")

    def __str__(self) -> str:
        """Return the function call string."""
        return str(self.value)

    def with_args(self, *args: Any) -> str:
        """Return function with arguments (for NEXTVAL)."""
        if isinstance(self.value, SequenceFunction):
            return self.value.with_args(*args)
        return str(self.value)


# ============================================================================
# CASCADE ACTIONS
# ============================================================================

class CascadeAction(Enum):
    """
    Foreign key cascade actions for referential integrity.

    :class: CascadeAction
    :synopsis: Enumeration of cascade actions for foreign key constraints
    """

    CASCADE = "CASCADE"      # Delete/update related records
    SET_NULL = "SET NULL"    # Set foreign key to NULL
    SET_DEFAULT = "SET DEFAULT"  # Set foreign key to default value
    RESTRICT = "RESTRICT"    # Prevent deletion/update if referenced
    NO_ACTION = "NO ACTION"  # Similar to RESTRICT but deferred


class RelationshipMultiplicity(Enum):
    """
    Relationship multiplicity types.

    :class: RelationshipMultiplicity
    :synopsis: Enumeration of relationship multiplicity constraints
    """

    MANY_TO_MANY = "MANY_MANY"
    MANY_TO_ONE = "MANY_ONE"
    ONE_TO_MANY = "ONE_MANY"
    ONE_TO_ONE = "ONE_ONE"


# ============================================================================
# DDL GENERATION CONSTANTS
# ============================================================================

class DDLConstants(StrEnum):
    """DDL generation constants."""

    # @@ STEP 1: Define DDL keywords
    CREATE_NODE_TABLE: Final[str] = "CREATE NODE TABLE"
    CREATE_REL_TABLE: Final[str] = "CREATE REL TABLE"
    CREATE_INDEX: Final[str] = "CREATE INDEX"
    ALTER_TABLE: Final[str] = "ALTER TABLE"
    DROP_TABLE: Final[str] = "DROP TABLE"
    DROP_INDEX: Final[str] = "DROP INDEX"

    # @@ STEP 2: Define DDL clauses
    PRIMARY_KEY: Final[str] = "PRIMARY KEY"
    FOREIGN_KEY: Final[str] = "FOREIGN KEY"
    REFERENCES: Final[str] = "REFERENCES"
    UNIQUE: Final[str] = "UNIQUE"
    NOT_NULL: Final[str] = "NOT NULL"
    DEFAULT: Final[str] = "DEFAULT"
    CHECK: Final[str] = "CHECK"
    ON_DELETE: Final[str] = "ON DELETE"
    ON_UPDATE: Final[str] = "ON UPDATE"
    CASCADE: Final[str] = "CASCADE"
    SET_NULL: Final[str] = "SET NULL"
    RESTRICT: Final[str] = "RESTRICT"
    NO_ACTION: Final[str] = "NO ACTION"

    # @@ STEP 3: Define relationship keywords
    FROM: Final[str] = "FROM"
    TO: Final[str] = "TO"
    MANY_TO_ONE: Final[str] = "MANY_ONE"
    ONE_TO_MANY: Final[str] = "ONE_MANY"
    MANY_TO_MANY: Final[str] = "MANY_MANY"
    ONE_TO_ONE: Final[str] = "ONE_ONE"

    # @@ STEP 3.1: Define REL TABLE GROUP keywords for multi-node relationships
    CREATE_REL_TABLE_GROUP: Final[str] = "CREATE REL TABLE GROUP"
    REL_TABLE_GROUP_FROM: Final[str] = "FROM"
    REL_TABLE_GROUP_TO: Final[str] = "TO"

    # @@ STEP 3.2: Define relationship field names that should be excluded from queries
    REL_FROM_NODE_FIELD: Final[str] = "from_node"
    REL_TO_NODE_FIELD: Final[str] = "to_node"
    REL_FROM_NODE_PK_FIELD: Final[str] = "_priv_from_node_pk"
    REL_TO_NODE_PK_FIELD: Final[str] = "_priv_to_node_pk"
    # @@ STEP 3.3: Default primary key field name for unresolved string node refs
    DEFAULT_PK_FIELD_NAME: Final[str] = "id"


    # @@ STEP 4: Define formatting constants
    STATEMENT_SEPARATOR: Final[str] = ";"
    FIELD_SEPARATOR: Final[str] = ", "
    NEWLINE: Final[str] = "\n"
    INDENT: Final[str] = "    "
    COMMENT_PREFIX: Final[str] = "-- "


# ============================================================================
# CYPHER QUERY CONSTANTS
# ============================================================================

class CypherConstants:
    """Cypher query language constants."""

    # @@ STEP 1: Define Cypher keywords
    MATCH: Final[str] = "MATCH"
    WHERE: Final[str] = "WHERE"
    RETURN: Final[str] = "RETURN"
    CREATE: Final[str] = "CREATE"
    MERGE: Final[str] = "MERGE"
    DELETE: Final[str] = "DELETE"
    SET: Final[str] = "SET"
    REMOVE: Final[str] = "REMOVE"
    WITH: Final[str] = "WITH"
    UNION: Final[str] = "UNION"
    UNION_ALL: Final[str] = "UNION ALL"
    ORDER_BY: Final[str] = "ORDER BY"
    SKIP: Final[str] = "SKIP"
    LIMIT: Final[str] = "LIMIT"
    DISTINCT: Final[str] = "DISTINCT"
    DETACH_DELETE: Final[str] = "DETACH DELETE"

    # @@ STEP 2: Define Cypher operators
    AND: Final[str] = "AND"
    OR: Final[str] = "OR"
    NOT: Final[str] = "NOT"
    XOR: Final[str] = "XOR"
    IN: Final[str] = "IN"
    IS_NULL: Final[str] = "IS NULL"
    IS_NOT_NULL: Final[str] = "IS NOT NULL"
    STARTS_WITH: Final[str] = "STARTS WITH"
    ENDS_WITH: Final[str] = "ENDS WITH"
    CONTAINS: Final[str] = "CONTAINS"
    LIKE: Final[str] = "LIKE"
    BETWEEN: Final[str] = "BETWEEN"
    EXISTS: Final[str] = "EXISTS"

    # @@ STEP 3: Define comparison operators
    EQ: Final[str] = "="
    NEQ: Final[str] = "<>"
    LT: Final[str] = "<"
    LTE: Final[str] = "<="
    GT: Final[str] = ">"
    GTE: Final[str] = ">="

    # @@ STEP 4: Define aggregate functions
    COUNT: Final[str] = "COUNT"
    SUM: Final[str] = "SUM"
    AVG: Final[str] = "AVG"
    MIN: Final[str] = "MIN"
    MAX: Final[str] = "MAX"
    COLLECT: Final[str] = "COLLECT"
    COLLECT_LIST: Final[str] = "COLLECT_LIST"
    COLLECT_SET: Final[str] = "COLLECT_SET"

    # @@ STEP 5: Define join types
    INNER_JOIN: Final[str] = "INNER JOIN"
    LEFT_JOIN: Final[str] = "LEFT JOIN"
    RIGHT_JOIN: Final[str] = "RIGHT JOIN"
    FULL_JOIN: Final[str] = "FULL JOIN"
    CROSS_JOIN: Final[str] = "CROSS JOIN"

    # @@ STEP 6: Define order directions
    ASC: Final[str] = "ASC"
    DESC: Final[str] = "DESC"

    # @@ STEP 7: Define grouping keywords
    GROUP_BY: Final[str] = "GROUP BY"
    HAVING: Final[str] = "HAVING"

    # @@ STEP 8: Define parameter prefix
    PARAM_PREFIX: Final[str] = "$"
    PARAM_SEPARATOR: Final[str] = "_"

# ============================================================================
# QUERY RETURN ALIAS CONSTANTS
# ============================================================================

class QueryReturnAliasConstants:
    """Constants for standardized aliases in query RETURN clauses."""

    FROM_ENDPOINT: Final[str] = "__from_endpoint__"
    TO_ENDPOINT: Final[str] = "__to_endpoint__"
    FROM_ID: Final[str] = "__from_id__"
    TO_ID: Final[str] = "__to_id__"


# ============================================================================
# MODEL METADATA CONSTANTS
# ============================================================================

class ModelMetadataConstants:
    """Model metadata attribute constants."""

    # @@ STEP 1: Define node metadata attributes
    KUZU_NODE_NAME: Final[str] = "__kuzu_node_name__"
    IS_KUZU_NODE: Final[str] = "__is_kuzu_node__"
    KUZU_NODE_LABELS: Final[str] = "__kuzu_node_labels__"
    KUZU_COMPOUND_INDEXES: Final[str] = "__kuzu_compound_indexes__"

    # @@ STEP 2: Define relationship metadata attributes
    KUZU_REL_NAME: Final[str] = "__kuzu_rel_name__"
    IS_KUZU_RELATIONSHIP: Final[str] = "__is_kuzu_relationship__"
    KUZU_REL_DIRECTION: Final[str] = "__kuzu_rel_direction__"
    KUZU_REL_MULTIPLICITY: Final[str] = "__kuzu_rel_multiplicity__"

    # @@ STEP 3: Define field metadata attributes
    KUZU_FIELDS: Final[str] = "__kuzu_fields__"
    KUZU_PRIMARY_KEY: Final[str] = "__kuzu_primary_key__"
    KUZU_UNIQUE_FIELDS: Final[str] = "__kuzu_unique_fields__"
    KUZU_INDEXED_FIELDS: Final[str] = "__kuzu_indexed_fields__"
    KUZU_FIELD_METADATA: Final[str] = "kuzu_metadata"

    # @@ STEP 4: Define validation attributes
    KUZU_VALIDATED: Final[str] = "__kuzu_validated__"
    KUZU_VALIDATION_ERRORS: Final[str] = "__kuzu_validation_errors__"


# ============================================================================
# NODE BASE CLASS CONSTANTS
# ============================================================================

class NodeBaseConstants:
    """Constants for the KuzuNodeBase class and node-specific functionality."""

    # @@ STEP 1: Define node base class identification
    IS_KUZU_NODE_BASE: Final[str] = "__is_kuzu_node_base__"

    # @@ STEP 2: Define node-specific error messages
    NOT_A_NODE_INSTANCE: Final[str] = "Expected a KuzuNodeBase instance or primary key value, got: {}"
    INVALID_NODE_REFERENCE: Final[str] = "Invalid node reference: must be a KuzuNodeBase instance or a valid primary key value"

    # @@ STEP 3: Define node validation messages
    NODE_MISSING_DECORATOR: Final[str] = "Node class '{}' must be decorated with @kuzu_node"
    NODE_MISSING_PRIMARY_KEY: Final[str] = "Node class '{}' must have at least one primary key field"


# ============================================================================
# ERROR MESSAGE CONSTANTS
# ============================================================================

class ErrorMessages:
    """Error message constants."""

    # @@ STEP 1: Define connection errors
    CONNECTION_FAILED: Final[str] = "Failed to establish database connection"
    CONNECTION_CLOSED: Final[str] = "Database connection is closed"
    CONNECTION_TIMEOUT: Final[str] = "Database connection timed out"

    # @@ STEP 2: Define model errors
    MODEL_NOT_REGISTERED: Final[str] = "Model {model_name} is not registered"
    MODEL_VALIDATION_FAILED: Final[str] = "Model validation failed: {errors}"
    INVALID_MODEL_TYPE: Final[str] = "Invalid model type: expected {expected}, got {actual}"
    MISSING_PRIMARY_KEY: Final[str] = "Model {model_name} is missing a primary key"
    DUPLICATE_PRIMARY_KEY: Final[str] = "Model {model_name} has multiple primary keys"

    # @@ STEP 3: Define field errors
    FIELD_NOT_FOUND: Final[str] = "Field {field_name} not found in model {model_name}"
    INVALID_FIELD_TYPE: Final[str] = "Invalid field type for {field_name}: {error}"
    MISSING_REQUIRED_FIELD: Final[str] = "Required field {field_name} is missing"
    INVALID_FIELD_VALUE: Final[str] = "Invalid value for field {field_name}: {value}"

    # @@ STEP 3.1: Define UUID-specific field errors
    UUID_STRING_NOT_ALLOWED: Final[str] = (
        "UUID field '{field_name}' in {model_name} must be a UUID object, not a string. "
        "Got: {value} ({type_name}). Use uuid.UUID('{value}') to create a proper UUID object."
    )

    # @@ STEP 4: Define relationship errors
    RELATIONSHIP_NOT_FOUND: Final[str] = "Relationship {rel_name} not found"
    INVALID_RELATIONSHIP_NODES: Final[str] = "Invalid nodes for relationship {rel_name}"
    CIRCULAR_RELATIONSHIP: Final[str] = "Circular relationship detected: {path}"

    # @@ STEP 5: Define query errors
    QUERY_BUILD_FAILED: Final[str] = "Failed to build query: {error}"
    QUERY_EXECUTION_FAILED: Final[str] = "Query execution failed: {error}"
    INVALID_QUERY_SYNTAX: Final[str] = "Invalid query syntax: {query}"
    QUERY_TIMEOUT: Final[str] = "Query execution timed out after {timeout} seconds"

    # @@ STEP 6: Define session errors
    SESSION_CLOSED: Final[str] = "Session is closed"
    TRANSACTION_ALREADY_ACTIVE: Final[str] = "Transaction is already active"
    NO_ACTIVE_TRANSACTION: Final[str] = "No active transaction"
    TRANSACTION_COMMIT_FAILED: Final[str] = "Failed to commit transaction: {error}"
    TRANSACTION_ROLLBACK_FAILED: Final[str] = "Failed to rollback transaction: {error}"

    # @@ STEP 7: Define validation errors
    VALIDATION_FAILED: Final[str] = "Validation failed: {errors}"
    TYPE_VALIDATION_FAILED: Final[str] = "Type validation failed for {field}: expected {expected}, got {actual}"
    CONSTRAINT_VIOLATION: Final[str] = "Constraint violation: {constraint}"

    # @@ STEP 8: Define foreign key validation errors
    FOREIGN_KEY_VALIDATION_FAILED: Final[str] = "Foreign key validation failed for {model_name}: {errors}"

    # @@ STEP 8: Define generic errors
    INVALID_ARGUMENT: Final[str] = "Invalid argument: {argument}"
    NOT_IMPLEMENTED: Final[str] = "Feature not implemented: {feature}"
    INTERNAL_ERROR: Final[str] = "Internal error: {error}"

# ============================================================================
# PERFORMANCE CONSTANTS
# ============================================================================

class PerformanceConstants:
    """Performance tuning constants."""

    # @@ STEP 1: Define cache settings
    CACHE_SIZE: Final[int] = 1000
    CACHE_TTL: Final[int] = 3600  # 1 hour in seconds
    CACHE_MAX_AGE: Final[int] = 86400  # 24 hours in seconds

    # @@ STEP 2: Define pool settings (fixed constants; no env)
    CONNECTION_POOL_SIZE: Final[int] = 10
    CONNECTION_POOL_MAX_OVERFLOW: Final[int] = 20
    CONNECTION_POOL_TIMEOUT: Final[int] = 30

    # @@ STEP 3: Define query optimization
    QUERY_CACHE_SIZE: Final[int] = 500
    QUERY_PLAN_CACHE_SIZE: Final[int] = 100
    STATISTICS_CACHE_TTL: Final[int] = 300  # 5 minutes

    # @@ STEP 4: Define batch processing
    BATCH_INSERT_SIZE: Final[int] = 1000
    BATCH_UPDATE_SIZE: Final[int] = 500
    BATCH_DELETE_SIZE: Final[int] = 500

    # @@ STEP 5: Define session optimization settings (fixed constants; no env)
    CONNECTION_REUSE_THRESHOLD: Final[int] = 5  # Reuse connection for N operations
    AUTOFLUSH_BATCH_SIZE: Final[int] = 100  # Batch size before forcing flush
    IDENTITY_MAP_INITIAL_SIZE: Final[int] = 256  # Initial identity map size
    METADATA_CACHE_SIZE: Final[int] = 500  # Cache size for model metadata


# ============================================================================
# DEFAULT VALUE CONSTANTS
# ============================================================================

class DefaultValueConstants:
    """Constants for default value handling."""

    # @@ STEP 1: Define SQL keywords
    NULL_KEYWORD: Final[str] = "NULL"
    TRUE_KEYWORD: Final[str] = "TRUE"
    FALSE_KEYWORD: Final[str] = "FALSE"

    # @@ STEP 2: Define default value prefixes
    DEFAULT_PREFIX: Final[str] = "DEFAULT"

    # @@ STEP 3: Define boolean representations
    BOOL_TRUE: Final[str] = "true"
    BOOL_FALSE: Final[str] = "false"

    # @@ STEP 4: Define string escaping
    QUOTE_CHAR: Final[str] = "'"
    ESCAPED_QUOTE: Final[str] = "''"


# ============================================================================
# RELATIONSHIP DIRECTION CONSTANTS
# ============================================================================

class RelationshipDirection:
    """Constants for relationship directions."""

    # @@ STEP 1: Define direction symbols
    FORWARD_ARROW: Final[str] = "->"
    BACKWARD_ARROW: Final[str] = "<-"
    BOTH_ARROW: Final[str] = "<->"

    # @@ STEP 2: Define direction names
    OUTGOING: Final[str] = "outgoing"
    INCOMING: Final[str] = "incoming"
    BOTH: Final[str] = "both"

    # @@ STEP 3: Define direction aliases
    FORWARD: Final[str] = "forward"
    BACKWARD: Final[str] = "backward"


# ============================================================================
# KUZU DATA TYPE CONSTANTS
# ============================================================================
class DDLMessageConstants:
    """Message templates used during DDL generation for warnings and info."""

    # Warnings related to relationship DDL generation
    WARN_UNKNOWN_FROM_NODE: Final[str] = (
        "Warning: Unknown FROM node referenced in "
        "relationship DDL: {}"
    )
    WARN_UNKNOWN_TO_NODE: Final[str] = (
        "Warning: Unknown TO node referenced in "
        "relationship DDL: {}"
    )
    WARN_DUPLICATE_REL_PAIR: Final[str] = (
        "Warning: Duplicate relationship pair (FROM {} TO {}) "
        "removed from emitted DDL"
    )

class KuzuDataType(StrEnum):
    """Constants for Kuzu data types."""

    # @@ STEP 1: Define integer types
    INT8: Final[str] = "INT8"
    INT16: Final[str] = "INT16"
    INT32: Final[str] = "INT32"
    INT64: Final[str] = "INT64"
    INT128: Final[str] = "INT128"
    UINT8: Final[str] = "UINT8"
    UINT16: Final[str] = "UINT16"
    UINT32: Final[str] = "UINT32"
    UINT64: Final[str] = "UINT64"

    # @@ STEP 2: Define floating point types
    FLOAT: Final[str] = "FLOAT"
    DOUBLE: Final[str] = "DOUBLE"
    DECIMAL: Final[str] = "DECIMAL"

    # @@ STEP 3: Define special numeric types
    SERIAL: Final[str] = "SERIAL"

    # @@ STEP 4: Define string types
    STRING: Final[str] = "STRING"

    # @@ STEP 5: Define boolean types
    BOOL: Final[str] = "BOOL"
    BOOLEAN: Final[str] = "BOOLEAN"

    # @@ STEP 6: Define temporal types
    DATE: Final[str] = "DATE"
    TIME: Final[str] = "TIME"
    TIMESTAMP: Final[str] = "TIMESTAMP"
    TIMESTAMP_NS: Final[str] = "TIMESTAMP_NS"
    TIMESTAMP_MS: Final[str] = "TIMESTAMP_MS"
    TIMESTAMP_SEC: Final[str] = "TIMESTAMP_SEC"
    TIMESTAMP_TZ: Final[str] = "TIMESTAMP_TZ"
    INTERVAL: Final[str] = "INTERVAL"

    # @@ STEP 7: Define binary types
    BLOB: Final[str] = "BLOB"

    # @@ STEP 8: Define identifier types
    UUID: Final[str] = "UUID"

    # @@ STEP 9: Define complex types
    ARRAY: Final[str] = "ARRAY"
    STRUCT: Final[str] = "STRUCT"
    MAP: Final[str] = "MAP"
    UNION: Final[str] = "UNION"

    # @@ STEP 10: Define node/relationship references
    NODE: Final[str] = "NODE"
    REL: Final[str] = "REL"



# ============================================================================
# CONSTRAINT CONSTANTS
# ============================================================================

class ConstraintConstants:
    """Constants for database constraints."""

    # @@ STEP 1: Define constraint types
    CHECK: Final[str] = "CHECK"
    UNIQUE: Final[str] = "UNIQUE"
    CONSTRAINT: Final[str] = "CONSTRAINT"

    # @@ STEP 2: Define index constants
    INDEX: Final[str] = "INDEX"
    UNIQUE_INDEX: Final[str] = "UNIQUE "
    INDEX_PREFIX: Final[str] = "idx"
    INDEX_SEPARATOR: Final[str] = "_"


# ============================================================================
# ARRAY TYPE CONSTANTS
# ============================================================================

class ArrayTypeConstants:
    """Constants for array type specifications."""

    # @@ STEP 1: Define array notation
    ARRAY_SUFFIX: Final[str] = "[]"

    # @@ STEP 2: Define element type separators
    ELEMENT_SEPARATOR: Final[str] = ", "


# ============================================================================
# QUERY FIELD CONSTANTS
# ============================================================================

class QueryFieldConstants:
    """Constants for query field operations."""
    PRIVATE_FIELD_PREFIX: Final[str] = "_"


# ============================================================================
# SESSION OPERATION CONSTANTS
# ============================================================================

class SessionOperationConstants:
    """Constants for session operations."""

    # @@ STEP 1: Define entity types
    NODE_ENTITY: Final[str] = "NODE"
    REL_ENTITY: Final[str] = "REL"

    # @@ STEP 2: Define operation messages
    NO_DDL_STATEMENTS: Final[str] = "No DDL statements to execute"
    SKIPPING_CREATED_NODE: Final[str] = "Skipping already created node table: {}"
    SKIPPING_CREATED_REL: Final[str] = "Skipping already created relationship table: {}"
    CREATED_NODE_TABLE: Final[str] = "Created node table: {}"
    CREATED_REL_TABLE: Final[str] = "Created relationship table: {}"
    NODE_TABLE_EXISTS: Final[str] = "Node table already exists: {}"
    REL_TABLE_EXISTS: Final[str] = "Relationship table already exists: {}"
    TABLE_ALREADY_EXISTS: Final[str] = "Table {} already exists"

    # @@ STEP 3: Define error patterns
    ALREADY_EXISTS_PATTERN: Final[str] = "already exists in catalog"
    BINDER_EXCEPTION_PATTERN: Final[str] = "Binder exception"


# ============================================================================
# JOIN PATTERN CONSTANTS
# ============================================================================

class JoinPatternConstants:
    """Constants for join pattern generation."""

    # @@ STEP 1: Define pattern templates
    OUTGOING_PATTERN: Final[str] = "({source})-[{rel_pattern}]->({target})"
    INCOMING_PATTERN: Final[str] = "({source})<-[{rel_pattern}]-({target})"
    BOTH_PATTERN: Final[str] = "({source})-[{rel_pattern}]-({target})"

    # @@ STEP 2: Define pattern prefixes
    OPTIONAL_MATCH_PREFIX: Final[str] = "OPTIONAL MATCH "
    MATCH_PREFIX: Final[str] = "MATCH "

    # @@ STEP 3: Define hop notation
    HOP_SEPARATOR: Final[str] = ".."
    HOP_PREFIX: Final[str] = "*"

    # @@ STEP 4: Define property notation
    PROPERTY_SEPARATOR: Final[str] = ", "
    PROPERTY_PREFIX: Final[str] = "$"


# ============================================================================
# REGISTRY RESOLUTION CONSTANTS
# ============================================================================

class RegistryResolutionConstants:
    """Constants for the deferred registry resolution system."""

    # @@ STEP 1: Define resolution states
    RESOLUTION_STATE_UNRESOLVED: Final[str] = "unresolved"
    RESOLUTION_STATE_RESOLVING: Final[str] = "resolving"
    RESOLUTION_STATE_RESOLVED: Final[str] = "resolved"
    RESOLUTION_STATE_ERROR: Final[str] = "error"

    # @@ STEP 2: Define resolution phases
    PHASE_REGISTRATION: Final[str] = "registration"
    PHASE_STRING_RESOLUTION: Final[str] = "string_resolution"
    PHASE_DEPENDENCY_ANALYSIS: Final[str] = "dependency_analysis"
    PHASE_TOPOLOGICAL_SORT: Final[str] = "topological_sort"
    PHASE_FINALIZED: Final[str] = "finalized"

    # @@ STEP 3: Define target model resolution types
    TARGET_TYPE_STRING: Final[str] = "string"
    TARGET_TYPE_CLASS: Final[str] = "class"
    TARGET_TYPE_CALLABLE: Final[str] = "callable"

    # @@ STEP 4: Define circular dependency handling
    CIRCULAR_DEPENDENCY_DETECTED: Final[str] = "circular_dependency_detected"
    SELF_REFERENCE_ALLOWED: Final[str] = "self_reference_allowed"

    # @@ STEP 5: Define resolution error types
    ERROR_TARGET_NOT_FOUND: Final[str] = "target_model_not_found"
    ERROR_CIRCULAR_DEPENDENCY: Final[str] = "circular_dependency_error"
    ERROR_INVALID_TARGET_TYPE: Final[str] = "invalid_target_type"
    ERROR_RESOLUTION_TIMEOUT: Final[str] = "resolution_timeout"


# ============================================================================
# VALIDATION MESSAGE CONSTANTS
# ============================================================================

class ValidationMessageConstants:
    """Constants for validation messages."""

    # @@ STEP 1: Define generic validation messages
    ROW_NOT_DICT: Final[str] = "Row is not a dict and has no to_dict method"
    QUERY_VALIDATION_FAILED: Final[str] = "Query failed validation. This likely indicates a malformed query. Result type: {}. Error: {}"

    # @@ STEP 2: Define field validation messages
    CANNOT_ACCESS_PRIVATE_FIELD: Final[str] = "Cannot access private field {}"
    INVALID_ORDER_BY_ARGUMENT: Final[str] = "Invalid order_by argument: {}"
    INVALID_SELECT_FIELD: Final[str] = "Invalid select field: {}"

    # @@ STEP 3: Define model validation messages
    MISSING_KUZU_REL_NAME: Final[str] = "Relationship class {} is not a decorated relationship - missing __kuzu_rel_name__ attribute"
    MISSING_KUZU_NODE_NAME: Final[str] = "Target model {} is not a decorated node - missing __kuzu_node_name__ attribute"
    MUST_INHERIT_RELATIONSHIP_BASE: Final[str] = "{} must inherit from KuzuRelationshipBase"

    # @@ STEP 4: Define query state validation
    CANNOT_UPDATE_FIELD: Final[str] = "Cannot update non-existent field '{}' in QueryState. Valid fields: {}"

    # @@ STEP 5: Define node type determination error messages
    NO_PRIMARY_KEY_FIELD: Final[str] = "Node class '{}' has no primary key field defined"
    DATABASE_QUERY_FAILED: Final[str] = "Database query failed for node type '{}' with primary key value '{}': {}"
    MODEL_ATTRIBUTE_ERROR: Final[str] = "Model class '{}' missing required attributes: {}"
    PARAMETER_BINDING_ERROR: Final[str] = "Failed to bind parameter for node type '{}' with value '{}': {}"
    RESULT_PARSING_ERROR: Final[str] = "Failed to parse query result for node type '{}': {}"


# ============================================================================
# RELATIONSHIP NODE TYPE QUERY CONSTANTS
# ============================================================================

class RelationshipNodeTypeQueryConstants:
    """Constants for relationship node type querying functionality."""

    # @@ STEP 1: Define query type identifiers
    QUERY_TYPE_FROM: Final[str] = "from"
    QUERY_TYPE_TO: Final[str] = "to"

    # @@ STEP 2: Define cache keys
    CACHE_KEY_FROM_TO_MAP: Final[str] = "from_to_map"
    CACHE_KEY_TO_FROM_MAP: Final[str] = "to_from_map"
    CACHE_KEY_FROM_TO_SINGLE: Final[str] = "from_to_single"
    CACHE_KEY_TO_FROM_SINGLE: Final[str] = "to_from_single"

    # @@ STEP 2.1: Adjacency storage keys for vectorized multi-node queries
    CACHE_KEY_ADJACENCY_DATA: Final[str] = "adjacency_data"
    ADJ_FROM_TO: Final[str] = "adj_from_to"
    ADJ_TO_FROM: Final[str] = "adj_to_from"
    FROM_LIST: Final[str] = "from_list"
    TO_LIST: Final[str] = "to_list"
    FROM_INDEX: Final[str] = "from_index"
    TO_INDEX: Final[str] = "to_index"

    # @@ STEP 3: Define error messages
    NO_RELATIONSHIP_PAIRS: Final[str] = "Relationship class '{}' has no relationship pairs defined"
    INVALID_NODE_TYPE: Final[str] = "Invalid node type '{}': must be a class, not {}"
    ABSTRACT_RELATIONSHIP_QUERY: Final[str] = (
        "Cannot query node types on abstract relationship class '{}'"
    )


# ============================================================================
# FOREIGN KEY VALIDATION CONSTANTS
# ============================================================================

class ForeignKeyValidationConstants:
    """Constants for foreign key validation system."""

    # @@ STEP 1: Define cache configuration
    CACHE_MAX_SIZE: Final[int] = 1000
    CACHE_KEY_SEPARATOR: Final[str] = ":"


# ============================================================================
# RELATIONSHIP NODE TYPE QUERY ERROR CONSTANTS
# ============================================================================

class RelationshipNodeTypeQueryErrorConstants:
    """Error constants for relationship node type queries."""

    ABSTRACT_RELATIONSHIP_QUERY: Final[str] = (
        "Cannot query node types on abstract relationship class '{}'"
    )
    INVALID_NODE_TYPE: Final[str] = "Invalid node type '{}': must be a class, not {}"


# ============================================================================
# EXPORT ALL CONSTANTS
# ============================================================================

__all__ = [
    "KuzuDefaultFunction",
    "DDLConstants",
    "DDLMessageConstants",
    "CypherConstants",
    "ModelMetadataConstants",
    "NodeBaseConstants",
    "ErrorMessages",
    "PerformanceConstants",
    "DefaultValueConstants",
    "RelationshipDirection",
    "KuzuDataType",
    "ConstraintConstants",
    "ArrayTypeConstants",
    "QueryFieldConstants",
    "SessionOperationConstants",
    "JoinPatternConstants",
    "ValidationMessageConstants",
    "QueryReturnAliasConstants",
    "RelationshipNodeTypeQueryConstants",
    "ForeignKeyValidationConstants",
    "RelationshipNodeTypeQueryErrorConstants",
]

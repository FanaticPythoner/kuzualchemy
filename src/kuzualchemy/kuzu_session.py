# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
Session management for Kuzu ORM with query execution and transaction support.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Tuple, Iterator, cast
from contextlib import contextmanager
from threading import RLock
from pathlib import Path
from collections import defaultdict
import ahocorasick

import polars as pl

from .kuzu_query import Query
from .constants import ValidationMessageConstants, QueryFieldConstants, ErrorMessages
from .connection_pool import get_shared_connection_pool
from .constants import PerformanceConstants

ModelType = TypeVar("ModelType")




class KuzuConnection:
    """Wrapper for Kuzu database connection using shared Database objects."""

    # Class-level Aho-Corasick automaton for efficient keyword matching
    _KEYWORD_AUTOMATON = None
    _KEYWORD_AUTOMATON_KEYWORDS = ['return', 'order', 'limit', 'union']
    _KEYWORD_AUTOMATON_KEYWORDS_TERMINATING = {'order', 'limit', 'union'}

    @classmethod
    def _get_keyword_automaton(cls):
        """Get or create the keyword automaton using pyahocorasick if available."""
        if cls._KEYWORD_AUTOMATON is None:

            # Use the optimized pyahocorasick library
            cls._KEYWORD_AUTOMATON = ahocorasick.Automaton()
            for keyword in cls._KEYWORD_AUTOMATON_KEYWORDS:
                cls._KEYWORD_AUTOMATON.add_word(keyword.lower(), keyword.lower())
            cls._KEYWORD_AUTOMATON.make_automaton()

        return cls._KEYWORD_AUTOMATON

    def __init__(self, db_path: Union[str, Path], read_only: bool = False, buffer_pool_size: int = 512 * 1024 * 1024, **kwargs):
        """Initialize connection to Kuzu database using connection pool."""
        # @@ STEP: Use connection pool for proper concurrent access with limited buffer pool size
        # || S.1: Get connection from shared Database object to avoid file locking issues
        # || S.2: Validate and limit buffer pool size to prevent massive memory allocation errors
        self.db_path = Path(db_path)
        self.read_only = read_only

        # Validate buffer_pool_size to prevent system crashes
        if buffer_pool_size is None or buffer_pool_size <= 0 or buffer_pool_size > 2**63:
            buffer_pool_size = 512 * 1024 * 1024  # Default to 512MB

        # Cap buffer pool size to reasonable maximum (2GB)
        max_buffer_size = 2 * 1024 * 1024 * 1024  # 2GB
        if buffer_pool_size > max_buffer_size:
            buffer_pool_size = max_buffer_size

        self.buffer_pool_size = buffer_pool_size
        # Acquire a shared connection pool instead of a dedicated connection to avoid
        # exhausting native resources under stress.
        self._pool = get_shared_connection_pool(
            self.db_path,
            read_only=read_only,
            max_connections=PerformanceConstants.CONNECTION_POOL_SIZE,
            buffer_pool_size=self.buffer_pool_size,
        )
        self._lock = RLock()
        self._closed = False

    def execute(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a Cypher query and normalize results."""
        with self._lock:
            if self._closed:
                raise RuntimeError(ErrorMessages.CONNECTION_CLOSED)
            conn = self._pool.get_connection()
            try:
                if parameters:
                    result = conn.execute(query, parameters)
                else:
                    result = conn.execute(query)
            finally:
                # Return the connection immediately to the pool to keep the number of
                # active native connections bounded, even if many logical sessions exist.
                self._pool.return_connection(conn)

            # @@ STEP: Normalize Kuzu result format
            # || S.1: Kuzu returns each row as a list of values
            # || S.2: Need to map to dict based on RETURN clause
            normalized = []
            
            # Parse RETURN clause to get column names
            return_aliases = self._parse_return_clause(query)
            
            for row in result:
                if isinstance(row, list):
                    # Row is a list of values, map to dict using return aliases
                    if len(return_aliases) == len(row):
                        row_dict = {}
                        for i, alias in enumerate(return_aliases):
                            row_dict[alias] = row[i]
                        normalized.append(row_dict)
                    else:
                        # Fallback: single value
                        if len(row) == 1 and return_aliases:
                            normalized.append({return_aliases[0]: row[0]})
                        else:
                            normalized.append(row)
                else:
                    # Already a dict or other structure
                    normalized.append(row)
            
            return normalized
    
    def _parse_return_clause(self, query: str) -> List[str]:
        """
        Parse RETURN clause to extract column aliases using Aho-Corasick automaton.

        Efficiently finds RETURN keyword and terminating keywords (ORDER, LIMIT, UNION)
        using O(n + m + z) complexity instead of regex backtracking.

        Args:
            query: Cypher query string

        Returns:
            List of column aliases from RETURN clause
        """
        # Use Aho-Corasick automaton to find all keyword matches
        automaton = self._get_keyword_automaton()
        matches = []

        # Use pyahocorasick
        query_lower = query.lower()
        for end_pos, keyword in automaton.iter(query_lower):
            start_pos = end_pos - len(keyword) + 1
            matches.append((start_pos, keyword))

        if not matches:
            return []

        # Find RETURN keyword position
        return_pos = None
        return_end = None

        for pos, keyword in matches:
            if keyword == 'return':
                return_pos = pos
                return_end = pos + len(keyword)
                break

        if return_pos is None or return_end is None:
            return []

        # Find the next terminating keyword after RETURN
        clause_end = len(query)  # Default to end of query

        for pos, keyword in matches:
            if keyword in KuzuConnection._KEYWORD_AUTOMATON_KEYWORDS_TERMINATING and pos > return_end:
                clause_end = pos
                break

        # Extract RETURN clause content
        return_content = query[return_end:clause_end].strip()

        # Skip whitespace after RETURN keyword
        return_content = return_content.lstrip()

        if not return_content:
            return []

        # Parse column aliases from RETURN clause
        parts = return_content.split(',')
        aliases = []

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Handle "expr AS alias" or just "expr" (case-insensitive)
            part_lower = part.lower()
            as_index = part_lower.find(' as ')

            if as_index != -1:
                # Extract alias after AS keyword
                alias = part[as_index + 4:].strip()
            else:
                # Use entire expression as alias
                alias = part.strip()

            if alias:
                aliases.append(alias)

        return aliases
    
    def close(self) -> None:
        """Mark this logical connection as closed; further executes will fail."""
        with self._lock:
            self._closed = True


class KuzuSession:
    """
    Session for executing queries and managing transactions.
    Provides SQLAlchemy-like interface for Kuzu database operations.
    """
    
    def __init__(
        self,
        connection: Optional[KuzuConnection] = None,
        db_path: Optional[Union[str, Path]] = None,
        read_only: bool = False,
        buffer_pool_size: int = 512 * 1024 * 1024,
        autoflush: bool = True,
        autocommit: bool = False,
        expire_on_commit: bool = True,
        bulk_insert_threshold: int = 10,
        bulk_batch_size: int = 10000,
        **kwargs
    ):
        """
        Initialize a Kuzu session.
        
        Args:
            connection: Existing connection to use
            db_path: Path to database if creating new connection
            autoflush: Whether to auto-flush before queries
            autocommit: Whether to auto-commit after operations
            expire_on_commit: Whether to expire objects after commit
        """
        if connection:
            self._conn = connection
            self._owns_connection = False
        elif db_path:
            self._conn = KuzuConnection(db_path, read_only=read_only, buffer_pool_size=buffer_pool_size, **kwargs)
            self._owns_connection = True
        else:
            raise ValueError("Either connection or db_path must be provided")
        self.autoflush = autoflush
        self.autocommit = autocommit
        self.expire_on_commit = expire_on_commit
        self.bulk_insert_threshold = bulk_insert_threshold
        self.bulk_batch_size = bulk_batch_size
        self._dirty = set()
        self._new = set()
        self._deleted = set()
        self._identity_map: Dict[Tuple[Type, Any], Any] = {}
        self._flushing = False
    
    
    def query(
        self,
        model_class: Type[ModelType],
        alias: str = "n") -> Query[ModelType]:
        """
        Create a query for a model class.
        
        Args:
            model_class: The model class to query
            alias: Alias for the model in the query
            
        Returns:
            Query object for building and executing queries
        """
        return Query(model_class, session=self, alias=alias)
    
    def execute(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a raw Cypher query.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result dictionaries
        """
        if self.autoflush and not self._flushing:
            self.flush()
        
        result = self._conn.execute(query, parameters)
        
        # Convert Kuzu result to list of dicts
        rows = []
        if result:
            # @@ STEP: Check if this is a DDL statement
            # || S.1: DDL statements don't return row data
            query_upper = query.strip().upper()
            is_ddl = any(query_upper.startswith(ddl) for ddl in [
                'CREATE', 'DROP', 'ALTER', 'COPY'
            ])
            
            if is_ddl:
                # DDL statements return status, not rows
                return []
            
            try:
                for row in result:
                    if hasattr(row, '__dict__'):
                        rows.append(row.__dict__)
                    elif hasattr(row, 'to_dict') and callable(row.to_dict):
                        rows.append(row.to_dict())
                    else:
                        try:
                            rows.append(dict(row))
                        except (TypeError, ValueError) as conv_err:
                            # @@ STEP: Handle UNION query results that return tuples/lists
                            # || S.1: UNION queries may return results in tuple format
                            try:
                                # || S.2: Try to convert tuple/list to dict using column names
                                if hasattr(result, 'get_column_names'):
                                    column_names = result.get_column_names()
                                    if len(column_names) == len(row):
                                        row_dict = dict(zip(column_names, row))
                                        rows.append(row_dict)
                                        continue

                                # || S.3: Fallback - create generic column names
                                row_dict = {f"{QueryFieldConstants.COLUMN_PREFIX}{i}": val for i, val in enumerate(row)}
                                rows.append(row_dict)
                            except Exception:
                                # || S.4: Last resort - raise original error
                                raise ValueError(ValidationMessageConstants.ROW_NOT_DICT) from conv_err
            except TypeError as e:
                # Result is not iterable - THIS IS AN ERROR
                raise RuntimeError(
                    ValidationMessageConstants.QUERY_VALIDATION_FAILED.format(type(result).__name__, e)
                ) from e
        
        return rows
    
    def add(self, instance: Any) -> None:
        """
        Add an instance to the session for insertion.
        
        Args:
            instance: Model instance to add
        """
        self._new.add(instance)
        
        if self.autocommit:
            self.commit()
    
    def add_all(self, instances: List[Any]) -> None:
        """
        Add multiple instances to the session.

        Args:
            instances: List of model instances to add
        """
        for instance in instances:
            self.add(instance)

    def bulk_insert(self, instances: List[Any], batch_size: Optional[int] = None) -> None:
        """
        Perform fast bulk insert using Polars DataFrames and Kuzu's COPY FROM.

        This method is orders of magnitude faster than individual add() calls
        for large datasets (1000+ records).

        Args:
            instances: List of model instances to bulk insert
            batch_size: Number of records per batch (uses session default if None)
        """
        if not instances:
            return

        # Use session default batch size if not specified
        effective_batch_size = batch_size if batch_size is not None else self.bulk_batch_size

        # Group instances by model type for efficient batch processing
        model_groups = defaultdict(list)

        for instance in instances:
            model_class = type(instance)
            model_groups[model_class].append(instance)

        # Process each model type separately
        for model_class, model_instances in model_groups.items():
            # Check if this is a multi-pair relationship
            if (hasattr(model_class, '__kuzu_rel_name__') or
                hasattr(model_class, '__kuzu_relationship_name__')):
                # Check if it has multiple pairs
                pairs = getattr(model_class, '__kuzu_relationship_pairs__', [])
                if len(pairs) > 1:
                    # Multi-pair relationships cannot use bulk insert
                    # Fall back to individual inserts
                    for instance in model_instances:
                        self._insert_instance(instance)
                    continue

            self._bulk_insert_model_type(model_class, model_instances, effective_batch_size)

    def _bulk_insert_model_type(self, model_class: Type[Any], instances: List[Any], batch_size: int) -> None:
        """
        Bulk insert instances of a single model type using Polars DataFrame.

        Args:
            model_class: The model class type
            instances: List of instances of the same model type
            batch_size: Number of records per batch
        """
        for i in range(0, len(instances), batch_size):
            batch = instances[i:i + batch_size]
            self._process_batch_with_polars(model_class, batch)

    def _process_batch_with_polars(self, model_class: Type[Any], instances: List[Any]) -> None:
        """
        Process a batch of instances using Polars DataFrame and Kuzu's COPY FROM.

        Args:
            model_class: The model class type
            instances: List of instances to process
        """
        if not instances:
            return

        # Convert instances to dictionary format
        data_dict = {}
        df = None

        from .kuzu_orm import KuzuRelationshipBase
        is_relationship = issubclass(model_class, KuzuRelationshipBase)

        if is_relationship:
            # For relationships, we need to include from_node_pk and to_node_pk
            # Use class attribute instead of instance attribute to avoid deprecation warning
            field_names = list(model_class.model_fields.keys())

            # Add node reference columns for relationships
            data_dict['from_node_pk'] = []
            data_dict['to_node_pk'] = []

            # Initialize property columns
            for field_name in field_names:
                data_dict[field_name] = []

            # Extract data from relationship instances
            for instance in instances:
                # Add node references
                data_dict['from_node_pk'].append(instance.from_node_pk)
                data_dict['to_node_pk'].append(instance.to_node_pk)

                # Add relationship properties with proper type conversion
                instance_data = instance.model_dump()
                for field_name in field_names:
                    value = instance_data.get(field_name)
                    # Convert datetime/date objects to proper string format for Kuzu
                    if hasattr(value, 'isoformat'):
                        # datetime or date object
                        value = value.isoformat()
                    elif (hasattr(value, '__class__') and
                          'KuzuDefaultFunction' in str(value.__class__)):
                        # Handle KuzuDefaultFunction objects - generate actual values for bulk insert
                        # COPY FROM doesn't support DEFAULT functions, so we must generate values
                        value = self._generate_default_function_value(value)
                    data_dict[field_name].append(value)
        else:
            # For nodes, use regular field extraction
            # Use class attribute instead of instance attribute to avoid deprecation warning
            field_names = list(model_class.model_fields.keys())

            # Initialize columns
            for field_name in field_names:
                data_dict[field_name] = []

            # Extract data from instances with proper type conversion
            for instance in instances:
                instance_data = instance.model_dump()
                for field_name in field_names:
                    value = instance_data.get(field_name)
                    # Convert datetime/date objects to proper string format for Kuzu
                    if hasattr(value, 'isoformat'):
                        # datetime or date object
                        value = value.isoformat()
                    elif (hasattr(value, '__class__') and
                          'KuzuDefaultFunction' in str(value.__class__)):
                        # Handle KuzuDefaultFunction objects - generate actual values for bulk insert
                        # COPY FROM doesn't support DEFAULT functions, so we must generate values
                        value = self._generate_default_function_value(value)
                    data_dict[field_name].append(value)

        try:
            # Create Polars DataFrame
            df = pl.DataFrame(data_dict)

            # Clear data_dict to free memory immediately
            data_dict.clear()
            del data_dict

            # Determine table name
            if hasattr(model_class, '__kuzu_node_name__'):
                table_name = model_class.__kuzu_node_name__
            elif hasattr(model_class, '__kuzu_rel_name__'):
                table_name = model_class.__kuzu_rel_name__
            elif hasattr(model_class, '__kuzu_relationship_name__'):
                table_name = model_class.__kuzu_relationship_name__
            else:
                raise ValueError(f"Model {model_class.__name__} is not a registered node or relationship")

            # Execute COPY using the Polars DataFrame
            self._conn.execute(f"COPY {table_name} FROM $dataframe", {"dataframe": df})
        finally:
            # Force garbage collection to free memory immediately
            import gc
            gc.collect()

    def _generate_default_function_value(self, default_function: Any) -> str:
        """
        Generate actual values for KuzuDefaultFunction instances in bulk insert.

        COPY FROM doesn't support DEFAULT functions, so we must generate
        the actual values that the functions would produce.

        Uses the BulkInsertValueGeneratorRegistry for proper type-based dispatch.

        Args:
            default_function: KuzuDefaultFunction enum value

        Returns:
            Generated value as string in Kuzu-compatible format
        """
        from .kuzu_orm import BulkInsertValueGeneratorRegistry
        return BulkInsertValueGeneratorRegistry.generate_value(default_function)

    def create_relationship(
        self,
        relationship_class: Type[Any],
        from_node: Any,
        to_node: Any,
        **properties
    ) -> Any:
        """
        Create and add a relationship between two nodes.

        Args:
            relationship_class: The relationship class to create
            from_node: Source node instance or primary key
            to_node: Target node instance or primary key
            **properties: Additional relationship properties

        Returns:
            The created relationship instance
        """
        from .kuzu_orm import KuzuRelationshipBase

        if not issubclass(relationship_class, KuzuRelationshipBase):
            raise ValueError(f"{relationship_class.__name__} must inherit from KuzuRelationshipBase")

        relationship = relationship_class.create_between(from_node, to_node, **properties)
        self.add(relationship)
        return relationship
    
    def delete(self, instance: Any) -> None:
        """
        Mark an instance for deletion.
        
        Args:
            instance: Model instance to delete
        """
        self._deleted.add(instance)
        
        if instance in self._new:
            self._new.remove(instance)
        if instance in self._dirty:
            self._dirty.remove(instance)
        
        if self.autocommit:
            self.commit()
    
    def merge(self, instance: Any) -> Any:
        """
        Merge an instance into the session.
        
        Args:
            instance: Model instance to merge
            
        Returns:
            Merged instance
        """
        model_class = type(instance)
        
        # Get primary key value
        pk_value = self._get_primary_key(instance)
        if not pk_value:
            self.add(instance)
            return instance
        
        # Check identity map
        identity_key = (model_class, pk_value)
        if identity_key in self._identity_map:
            existing = self._identity_map[identity_key]
            # Update existing with new values
            for field_name, field_value in instance.model_dump().items():
                setattr(existing, field_name, field_value)
            self._dirty.add(existing)
            return existing
        else:
            # Add to identity map
            self._identity_map[identity_key] = instance
            self._dirty.add(instance)
            return instance
    
    def flush(self) -> None:
        """
        Flush pending changes to the database without committing.
        """
        if self._flushing:
            return

        self._flushing = True
        try:
            # Process new instances in correct order: nodes first, then relationships
            # This ensures that nodes exist before relationships try to reference them

            # Separate nodes and relationships
            nodes_to_insert = []
            relationships_to_insert = []

            for instance in self._new:
                model_class = type(instance)
                if hasattr(model_class, '__kuzu_node_name__'):
                    nodes_to_insert.append(instance)
                elif hasattr(model_class, '__kuzu_rel_name__'):
                    relationships_to_insert.append(instance)
                else:
                    # Unknown type - process immediately to maintain existing behavior
                    self._insert_instance(instance)

            # Use session configuration for bulk insert threshold
            # Process nodes first - use bulk insert if many instances
            if len(nodes_to_insert) >= self.bulk_insert_threshold:
                self.bulk_insert(nodes_to_insert)
            else:
                for instance in nodes_to_insert:
                    self._insert_instance(instance)

            # Then process relationships - use bulk insert if many instances
            if len(relationships_to_insert) >= self.bulk_insert_threshold:
                self.bulk_insert(relationships_to_insert)
            else:
                for instance in relationships_to_insert:
                    self._insert_instance(instance)

            # Process dirty instances
            for instance in self._dirty:
                self._update_instance(instance)

            # Process deleted instances
            for instance in self._deleted:
                self._delete_instance(instance)

            # Clear pending sets
            self._new.clear()
            self._dirty.clear()
            self._deleted.clear()
        finally:
            self._flushing = False
    
    def commit(self) -> None:
        """
        Flush all pending operations to the database.
        """
        self.flush()

        if self.expire_on_commit:
            self.expire_all()

    def rollback(self) -> None:
        """
        Clear all pending operations without executing them.
        """
        # Clear all pending changes
        self._new.clear()
        self._dirty.clear()
        self._deleted.clear()
        
        # Clear identity map
        self._identity_map.clear()
    
    def expire(self, instance: Any) -> None:
        """
        Expire an instance from the session.
        
        Args:
            instance: Instance to expire
        """
        pk_value = self._get_primary_key(instance)
        if pk_value:
            identity_key = (type(instance), pk_value)
            if identity_key in self._identity_map:
                del self._identity_map[identity_key]
    
    def expire_all(self) -> None:
        """
        Expire all instances from the session.
        """
        self._identity_map.clear()
    
    def close(self) -> None:
        """
        Close the session.
        """
        self.rollback()
        
        if self._owns_connection:
            self._conn.close()
    
    @contextmanager
    def begin(self) -> Iterator["KuzuSession"]:
        """
        Batch operations context manager.
        Execute multiple operations and flush them together.
        """
        original_autocommit = self.autocommit
        self.autocommit = False
        try:
            yield self
            self.commit()
        except (RuntimeError, ValueError, TypeError) as e:
            self.rollback()
            raise e
        except Exception as e:
            self.rollback()
            raise RuntimeError(f"Unexpected error in transaction: {type(e).__name__}: {e}") from e
        finally:
            self.autocommit = original_autocommit
    
    @contextmanager
    def begin_nested(self) -> Iterator["KuzuSession"]:
        """
        Nested batch operations context manager.
        Note: Kuzu doesn't support savepoints, so this is a best-effort simulation.
        """
        # Save current state
        saved_new = set(self._new)
        saved_dirty = set(self._dirty)
        saved_deleted = set(self._deleted)
        
        try:
            yield self
        except (RuntimeError, ValueError, TypeError) as e:
            # Restore previous state
            self._new = saved_new
            self._dirty = saved_dirty
            self._deleted = saved_deleted
            raise e
        except Exception as e:
            # Restore previous state
            self._new = saved_new
            self._dirty = saved_dirty
            self._deleted = saved_deleted
            raise RuntimeError(f"Unexpected error in nested session: {type(e).__name__}: {e}") from e
    
    def _get_primary_key(self, instance: Any) -> Optional[Any]:
        """Get primary key value from an instance."""
        model_class = type(instance)
        
        if hasattr(model_class, 'get_primary_key_fields'):
            get_pk_fields = model_class.get_primary_key_fields
            if not callable(get_pk_fields):
                raise ValueError(
                    f"Model {model_class.__name__}.get_primary_key_fields exists but is not callable"
                )
            pk_fields = cast(List[str], get_pk_fields())

            if len(pk_fields) == 1:
                pk_field = pk_fields[0]
                if not hasattr(instance, pk_field):
                    raise ValueError(
                        f"Primary key field '{pk_field}' not found on {model_class.__name__} instance"
                    )
                # @@ STEP 1: Check __dict__ first for performance (avoids descriptor overhead)
                value = instance.__dict__.get(pk_field)
                
                # @@ STEP 2: If not in __dict__, try getattr for properties/descriptors
                # || S.S.1: Only call getattr if we know the attribute exists
                if value is None and hasattr(instance, pk_field):
                    # || S.S.2: Safe to call getattr since hasattr confirmed existence
                    value = getattr(instance, pk_field)
                
                # @@ STEP 3: Validate that we got a non-None value
                if value is None:
                    # || S.S.3: Explicit error with detailed context
                    raise ValueError(
                        f"Primary key field '{pk_field}' is None on {model_class.__name__} instance. "
                        f"The field exists but has no value. Instance must have a non-None primary key."
                    )
                return value

            elif len(pk_fields) > 1:
                # Composite key - ALL fields must exist and be non-None
                values = []
                for field in pk_fields:
                    if not hasattr(instance, field):
                        raise ValueError(
                            f"Composite key field '{field}' not found on {model_class.__name__} instance"
                        )
                    # @@ STEP 1: Check __dict__ first for performance
                    value = instance.__dict__.get(field)
                    
                    # @@ STEP 2: If not in __dict__, try getattr for properties/descriptors
                    # || S.S.1: Only call getattr if we know the attribute exists
                    if value is None and hasattr(instance, field):
                        # || S.S.2: Safe to call getattr since hasattr confirmed existence
                        value = getattr(instance, field)
                    
                    # @@ STEP 3: Validate that we got a non-None value
                    if value is None:
                        # || S.S.3: Explicit error with detailed context
                        raise ValueError(
                            f"Composite key field '{field}' is None on {model_class.__name__} instance. "
                            f"The field exists but has no value. All composite key fields must be non-None."
                        )
                    values.append(value)
                return tuple(values)
        
        # Try common primary key field names
        for field_name in ('id', 'uid', 'uuid', 'pk'):
            if hasattr(instance, field_name):
                value = getattr(instance, field_name)
                if value is not None:
                    return value
        
        # No primary key found - THIS IS AN ERROR
        raise ValueError(
            f"Cannot determine primary key for {model_class.__name__} instance. "
            f"Instance must have a primary key field or implement get_primary_key_fields() method."
        )
    
    def _insert_instance(self, instance: Any) -> None:
        """Insert a new instance into the database."""
        model_class = type(instance)

        # Check if it's a node or relationship
        if hasattr(model_class, '__kuzu_node_name__'):
            self._insert_node_instance(instance)
        elif hasattr(model_class, '__kuzu_rel_name__'):
            self._insert_relationship_instance(instance)
        else:
            raise ValueError(
                f"Model {model_class.__name__} is not a registered node or relationship - "
                f"missing __kuzu_node_name__ or __kuzu_rel_name__ attribute"
            )

    def _insert_node_instance(self, instance: Any) -> None:
        """Insert a node instance into the database."""
        model_class = type(instance)
        node_name = model_class.__kuzu_node_name__

        # Build CREATE query
        properties = instance.model_dump(exclude_unset=True)

        if not properties:
            query = f"CREATE (:{node_name})"
            self._conn.execute(query)
        else:
            prop_str = ", ".join(f"{k}: ${k}" for k in properties.keys())
            query = f"CREATE (:{node_name} {{{prop_str}}})"
            self._conn.execute(query, properties)

        # Add to identity map
        pk_value = self._get_primary_key(instance)
        if pk_value:
            self._identity_map[(model_class, pk_value)] = instance

    def _insert_relationship_instance(self, instance: Any) -> None:
        """Insert a relationship instance into the database."""
        from .kuzu_orm import KuzuRelationshipBase

        if not isinstance(instance, KuzuRelationshipBase):
            raise ValueError(f"Relationship instance must inherit from KuzuRelationshipBase")

        model_class = type(instance)
        rel_name = model_class.get_relationship_name()

        # Get from and to node primary keys
        from_pk = instance.from_node_pk
        to_pk = instance.to_node_pk

        if from_pk is None or to_pk is None:
            raise ValueError(
                f"Relationship {model_class.__name__} must have both from_node and to_node specified"
            )

        # Get relationship pairs to determine node types
        rel_pairs = instance.get_relationship_pairs()
        if not rel_pairs:
            raise ValueError(f"No relationship pairs defined for {model_class.__name__}")

        # For now, use the first pair (could be enhanced for multi-pair relationships)
        from_node_class = rel_pairs[0].from_node
        to_node_class = rel_pairs[0].to_node

        # Get node names - handle both class types and strings
        if isinstance(from_node_class, str):
            from_node_name = from_node_class
        else:
            from_node_name = from_node_class.__kuzu_node_name__

        if isinstance(to_node_class, str):
            to_node_name = to_node_class
        else:
            to_node_name = to_node_class.__kuzu_node_name__

        # Build relationship properties (exclude internal fields)
        properties = instance.model_dump(exclude_unset=True)
        # Remove internal fields that shouldn't be in the relationship
        internal_fields = {'_from_node', '_to_node', '_from_node_pk', '_to_node_pk'}
        properties = {k: v for k, v in properties.items() if k not in internal_fields}

        # Get primary key field names
        if isinstance(from_node_class, str):
            from_pk_field = "id"  # Default assumption for string node names
        else:
            from_pk_field = self._get_primary_key_field_name(from_node_class)

        if isinstance(to_node_class, str):
            to_pk_field = "id"  # Default assumption for string node names
        else:
            to_pk_field = self._get_primary_key_field_name(to_node_class)

        # Build Cypher query to create relationship
        if properties:
            prop_str = ", ".join(f"{k}: ${k}" for k in properties.keys())
            query = f"""
            MATCH (from_node:{from_node_name}), (to_node:{to_node_name})
            WHERE from_node.{from_pk_field} = $from_pk
              AND to_node.{to_pk_field} = $to_pk
            CREATE (from_node)-[:{rel_name} {{{prop_str}}}]->(to_node)
            """
            params = {**properties, 'from_pk': from_pk, 'to_pk': to_pk}
        else:
            query = f"""
            MATCH (from_node:{from_node_name}), (to_node:{to_node_name})
            WHERE from_node.{from_pk_field} = $from_pk
              AND to_node.{to_pk_field} = $to_pk
            CREATE (from_node)-[:{rel_name}]->(to_node)
            """
            params = {'from_pk': from_pk, 'to_pk': to_pk}

        self._conn.execute(query, params)

    def _get_primary_key_field_name(self, model_class: Type[Any]) -> str:
        """Get the primary key field name for a model class."""
        from .kuzu_orm import _kuzu_registry

        for field_name, field_info in model_class.model_fields.items():
            metadata = _kuzu_registry.get_field_metadata(field_info)
            if metadata and metadata.primary_key:
                return field_name
        raise ValueError(f"No primary key field found in {model_class.__name__}")
    
    def _update_instance(self, instance: Any) -> None:
        """Update an existing instance in the database."""
        model_class = type(instance)
        
        if not hasattr(model_class, '__kuzu_node_name__'):
            raise ValueError(
                f"Model {model_class.__name__} is not a registered node - "
                f"missing __kuzu_node_name__ attribute"
            )
        node_name = model_class.__kuzu_node_name__
        pk_value = self._get_primary_key(instance)
        
        if not pk_value:
            raise ValueError(f"Cannot update instance without primary key")
        
        # Build UPDATE query
        properties = instance.model_dump(exclude_unset=True)
        
        # Remove primary key from properties to update
        if hasattr(model_class, 'get_primary_key_fields'):
            get_pk_fields = model_class.get_primary_key_fields
            if not callable(get_pk_fields):
                raise ValueError(
                    f"Model {model_class.__name__}.get_primary_key_fields exists but is not callable"
                )
            pk_fields = cast(List[str], get_pk_fields())
        else:
            pk_fields = ['id']
        for pk_field in pk_fields:
            properties.pop(pk_field, None)
        
        if properties:
            set_clause = ", ".join(f"n.{k} = ${k}" for k in properties.keys())

            # Build WHERE clause for primary key
            if len(pk_fields) == 1:
                where_clause = f"n.{pk_fields[0]} = $pk_value"
                params = {**properties, "pk_value": pk_value}
            else:
                where_parts = [f"n.{field} = $pk_{i}" for i, field in enumerate(pk_fields)]
                where_clause = " AND ".join(where_parts)
                params = {**properties}
                for i, value in enumerate(pk_value):
                    params[f"pk_{i}"] = value
            
            query = f"MATCH (n:{node_name}) WHERE {where_clause} SET {set_clause}"
            self._conn.execute(query, params)
    
    def _delete_instance(self, instance: Any) -> None:
        """Delete an instance from the database."""
        model_class = type(instance)
        
        if not hasattr(model_class, '__kuzu_node_name__'):
            raise ValueError(
                f"Model {model_class.__name__} is not a registered node - "
                f"missing __kuzu_node_name__ attribute"
            )
        node_name = model_class.__kuzu_node_name__
        pk_value = self._get_primary_key(instance)
        
        if not pk_value:
            raise ValueError(f"Cannot delete instance without primary key")
        
        # Build DELETE query
        if hasattr(model_class, 'get_primary_key_fields'):
            get_pk_fields = model_class.get_primary_key_fields
            if not callable(get_pk_fields):
                raise ValueError(
                    f"Model {model_class.__name__}.get_primary_key_fields exists but is not callable"
                )
            pk_fields = cast(List[str], get_pk_fields())
        else:
            pk_fields = ['id']

        if len(pk_fields) == 1:
            where_clause = f"n.{pk_fields[0]} = $pk_value"
            params = {"pk_value": pk_value}
        else:
            where_parts = [f"n.{field} = $pk_{i}" for i, field in enumerate(pk_fields)]
            where_clause = " AND ".join(where_parts)
            params = {}
            for i, value in enumerate(pk_value):
                params[f"pk_{i}"] = value
        
        query = f"MATCH (n:{node_name}) WHERE {where_clause} DELETE n"
        self._conn.execute(query, params)
        
        # Remove from identity map
        identity_key = (model_class, pk_value)
        if identity_key in self._identity_map:
            del self._identity_map[identity_key]
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        _ = exc_val, exc_tb  # Mark as intentionally unused
        try:
            if exc_type is None:
                self.commit()
            else:
                self.rollback()
        finally:
            self.close()
        self.close()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<KuzuSession(autocommit={self.autocommit})>"


class KuzuTransaction:
    """Context manager for batched operations in Kuzu."""
    
    def __init__(self, session: "KuzuSession"):
        """Initialize transaction."""
        self.session = session
        self._original_autocommit = session.autocommit
    
    def __enter__(self):
        """Begin batched operations."""
        self.session.autocommit = False
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End batched operations."""
        _ = exc_val, exc_tb  # Mark as intentionally unused
        try:
            if exc_type is None:
                self.session.commit()
            else:
                self.session.rollback()
        finally:
            self.session.autocommit = self._original_autocommit


class SessionFactory:
    """Factory for creating sessions with consistent configuration."""
    
    def __init__(
        self,
        db_path: Union[str, Path],
        **default_kwargs
    ):
        """
        Initialize session factory.
        
        Args:
            db_path: Path to database
            **default_kwargs: Default session configuration
        """
        self.db_path = Path(db_path)
        self.default_kwargs = default_kwargs
    
    def create_session(self, **kwargs) -> KuzuSession:
        """
        Create a new session.
        
        Args:
            **kwargs: Override default configuration
            
        Returns:
            New KuzuSession instance
        """
        config = {**self.default_kwargs, **kwargs}
        return KuzuSession(db_path=self.db_path, **config)
    
    @contextmanager
    def session_scope(self, **kwargs):
        """
        Provide a transactional scope for a series of operations.
        
        Args:
            **kwargs: Override default configuration
            
        Yields:
            KuzuSession instance
        """
        session = self.create_session(**kwargs)
        try:
            yield session
            session.commit()
        except (RuntimeError, ValueError, TypeError) as e:
            session.rollback()
            raise e
        except Exception as e:
            session.rollback()
            raise RuntimeError(f"Unexpected error in session: {type(e).__name__}: {e}") from e
        finally:
            session.close()
    
    def __call__(self, **kwargs) -> KuzuSession:
        """Allow factory to be called directly."""
        return self.create_session(**kwargs)

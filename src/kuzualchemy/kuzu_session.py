# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
Session management for Kuzu ORM with query execution and transaction support.

This module provides the core KuzuSession class that manages database connections,
query execution, transaction handling, and ORM operations for the KuzuAlchemy framework.
It implements connection pooling, identity mapping, bulk operations, and comprehensive
error handling with precision.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Iterator, cast
from contextlib import contextmanager
from threading import RLock
from pathlib import Path
from collections import defaultdict, OrderedDict
import ahocorasick
import logging
import uuid
from enum import Enum
import time
import gc

import pyarrow as pa

from .kuzu_query import Query
from .constants import (
    ValidationMessageConstants,
    QueryFieldConstants,
    ErrorMessages,
    KuzuDataType,
    LoggingConstants,
    DatabaseConstants,
)
from .connection_pool import get_shared_connection_pool
from .constants import PerformanceConstants
from .constants import DDLConstants
from .kuzu_orm import get_node_by_name, KuzuRelationshipBase, get_registered_nodes

# TODO: TESTS ARE FALING SINCE I MADE THE BATCH SIZES DYNAMIC ACCORDING TO KUZU, EVEN THOUGH MY IMPLEM ENTATION FIXED A SHIT LOAD OF ISSUES IN OTHER PRODUCTION PROJECTS. FIX THE ISSUES RIGHT NOW HERE AND INT HE CONNECTION POOL OR WHEREEVER.

ModelType = TypeVar("ModelType")

logger = logging.getLogger(__name__)


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

    def __init__(self, db_path: Union[str, Path], read_only: bool = False, buffer_pool_size: int = DatabaseConstants.DEFAULT_BUFFER_POOL_SIZE, **kwargs):
        """
        Initialize connection to Kuzu database using connection pool.

        Args:
            db_path: Path to the Kuzu database file or directory
            read_only: Whether to open database in read-only mode
            buffer_pool_size: Size of buffer pool in bytes. When 0, Kuzu auto-selects (~80% RAM).
            **kwargs: Additional connection parameters (unused but maintained for compatibility)

        Raises:
            ConnectionError: If database connection cannot be established
            ValueError: If invalid parameters are provided
        """
        # @@ STEP: Use connection pool for proper concurrent access with limited buffer pool size
        # || S.1: Get connection from shared Database object to avoid file locking issues
        # || S.2: Validate and limit buffer pool size to prevent massive memory allocation errors
        self.db_path = Path(db_path)
        self.read_only = read_only

        # Normalize buffer_pool_size: allow 0 to delegate sizing to Kuzu (recommended)
        if buffer_pool_size is None or buffer_pool_size < 0 or buffer_pool_size > 2**63:
            buffer_pool_size = DatabaseConstants.DEFAULT_BUFFER_POOL_SIZE

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
        buffer_pool_size: int = DatabaseConstants.DEFAULT_BUFFER_POOL_SIZE,
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
        # Optional performance knob: force GC after batch operations (defaults to False)
        local_force_gc = bool(kwargs.pop("force_gc", False))

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
        self._force_gc = local_force_gc

        self._deleted = set()
        self._flushing = False

        # || S.1: Use IDENTITY_MAP_INITIAL_SIZE for better memory allocation
        # || Pre-allocating dictionary size reduces hash collisions
        # || and memory reallocations during runtime, improving O(1) lookup performance
        initial_size = PerformanceConstants.IDENTITY_MAP_INITIAL_SIZE
        self._identity_map: Dict[str, Any] = dict.fromkeys(range(initial_size))
        self._identity_map.clear()  # Clear keys but keep allocated space

        self._reused_connection = None
        self._connection_operation_count = 0

        # || S.1: Track pending operations count for batch-based autoflush
        # || Batching reduces I/O overhead by factor of batch_size
        self._pending_operations_count = 0
        self._autoflush_batch_size = PerformanceConstants.AUTOFLUSH_BATCH_SIZE

        # || S.1: Initialize metadata cache with bounded size to prevent memory leaks
        # || LRU cache with fixed size provides O(1) access
        # || while maintaining bounded memory usage
        self._metadata_cache: OrderedDict[str, Any] = OrderedDict()
        self._metadata_cache_size = PerformanceConstants.METADATA_CACHE_SIZE


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
        # || Only flush if we have pending operations and autoflush is enabled
        if self.autoflush and not self._flushing and self._has_pending_operations():
            self.flush()

        result = self._execute_with_connection_reuse(query, parameters)

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

    def _has_pending_operations(self) -> bool:
        """
        Check if there are pending operations that require flushing based on batch size.

        Justification:
        - Batch flushing reduces I/O operations by factor of batch_size
        - Optimal batch size balances memory usage vs I/O efficiency
        - Formula: flush_needed = pending_count >= batch_threshold

        Returns:
            bool: True if pending operations exceed batch threshold
        """
        # || S.1: Count total pending operations across all operation types
        total_pending = len(self._new) + len(self._dirty) + len(self._deleted)

        # || S.2: Update pending operations count for tracking
        self._pending_operations_count = total_pending

        # || S.3: Decision: flush when batch size is reached
        # || This optimizes I/O by batching operations together
        batch_threshold_reached = total_pending >= self._autoflush_batch_size

        # || S.4: Also flush if we have any operations and autoflush is enabled
        # || This ensures operations don't get stuck indefinitely
        has_any_operations = total_pending > 0

        return batch_threshold_reached or (has_any_operations and self.autoflush)

    def _execute_with_connection_reuse(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute query with connection reuse optimization and resilient backoff on buffer exhaustion.

        This method attempts a small, bounded retry with exponential backoff if Kuzu's buffer
        manager reports exhaustion ("No more frame groups can be added to the allocator").
        The retry path is intentionally tight (up to 3 attempts) and resets connection reuse
        state to encourage fresh allocation from the pool between attempts.
        """
        def is_buffer_exhaustion_error(err: BaseException) -> bool:
            msg = str(err).lower()
            return (
                "buffer manager exception" in msg
                or "no more frame groups" in msg
                or "out of memory" in msg
            )

        # Helper to run a single execution call using either reused or regular connection
        def _do_exec(use_reused: bool) -> Any:
            if use_reused and (self._reused_connection is not None and self._connection_operation_count < PerformanceConstants.CONNECTION_REUSE_THRESHOLD):
                result = self._reused_connection.execute(query, parameters)
                self._connection_operation_count += 1
                return result
            # Fallback to main connection
            result = self._conn.execute(query, parameters)
            self._reused_connection = self._conn
            self._connection_operation_count = 1
            return result

        # First attempt: try reused connection when eligible
        try:
            return _do_exec(use_reused=True)
        except (ConnectionError, OSError, ValueError, RuntimeError) as e:
            # If not a buffer exhaustion error, try a single fall back to normal execution
            if not is_buffer_exhaustion_error(e):
                logger.warning(
                    LoggingConstants.CONNECTION_REUSE_ERROR_MSG.format(
                        query=query[:100] + "..." if len(query) > 100 else query,
                        error=str(e)
                    )
                )
                try:
                    # Reset reuse state then execute normally once
                    self._reused_connection = None
                    self._connection_operation_count = 0
                    return _do_exec(use_reused=False)
                except Exception as e:
                    raise e
            # Buffer exhaustion path: bounded exponential backoff retries
            backoff_s = 0.05
            max_attempts = 3
            last_exc: BaseException = e
            # Reset reuse state to encourage a fresh pooled connection
            self._reused_connection = None
            self._connection_operation_count = 0
            for attempt in range(1, max_attempts + 1):
                time.sleep(backoff_s)
                if self._force_gc:
                    gc.collect()
                try:
                    return _do_exec(use_reused=False)
                except (ConnectionError, OSError, ValueError, RuntimeError) as e2:
                    last_exc = e2
                    if not is_buffer_exhaustion_error(e2):
                        # Different error type: stop retrying and raise immediately
                        raise
                    backoff_s = min(backoff_s * 2.0, 0.5)
            # All retries failed; re-raise the last exception
            raise last_exc
        except Exception as unexpected_error:
            # Unexpected error type: reset and re-raise
            logger.error(
                f"Unexpected error during connection reuse: {unexpected_error}. "
                f"Query: {query[:100] + '...' if len(query) > 100 else query}"
            )
            self._reused_connection = None
            self._connection_operation_count = 0
            raise

    def _reset_connection_reuse(self) -> None:
        """Reset connection reuse state."""
        # || Simply reset the reuse state - no need to return connection since it's the same as _conn
        self._reused_connection = None
        self._connection_operation_count = 0

    def _generate_identity_key(self, model_class: Type[Any], pk_value: Any) -> str:
        """Generate optimized identity key for identity map."""
        # || Use string concatenation instead of tuple for better performance
        # || This avoids tuple creation and hashing overhead
        return f"{model_class.__name__}:{pk_value}"

    def _get_metadata_cache_key(self, model_class: Type[Any], metadata_type: str) -> str:
        """
        Generate cache key for metadata.

        Args:
            model_class: The model class
            metadata_type: Type of metadata (e.g., 'fields', 'pk_fields', 'table_name')

        Returns:
            str: Cache key for the metadata
        """
        return f"{model_class.__name__}:{metadata_type}"

    def _get_cached_metadata(self, model_class: Type[Any], metadata_type: str) -> Optional[Any]:
        """
        Get cached metadata for a model class.

        Justification:
        - LRU cache provides O(1) access time for frequently used metadata
        - Bounded cache size prevents memory leaks
        - Cache hit rate improves with repeated model operations

        Args:
            model_class: The model class
            metadata_type: Type of metadata to retrieve

        Returns:
            Optional[Any]: Cached metadata or None if not found
        """
        cache_key = self._get_metadata_cache_key(model_class, metadata_type)

        if cache_key in self._metadata_cache:
            # Move to end (most recently used)
            value = self._metadata_cache.pop(cache_key)
            self._metadata_cache[cache_key] = value
            return value

        return None

    def _set_cached_metadata(self, model_class: Type[Any], metadata_type: str, metadata: Any) -> None:
        """
        Set cached metadata for a model class with LRU eviction.

        Justification:
        - LRU eviction maintains most frequently accessed metadata
        - Bounded size prevents unbounded memory growth
        - O(1) insertion and eviction operations

        Args:
            model_class: The model class
            metadata_type: Type of metadata to cache
            metadata: The metadata to cache
        """
        cache_key = self._get_metadata_cache_key(model_class, metadata_type)

        # Remove oldest entries if cache is full
        while len(self._metadata_cache) >= self._metadata_cache_size:
            # Remove least recently used (first item)
            self._metadata_cache.popitem(last=False)

        # Add new entry (most recently used)
        self._metadata_cache[cache_key] = metadata

    def _clear_metadata_cache(self) -> None:
        """Clear all cached metadata."""
        self._metadata_cache.clear()

    def add(self, instance: Any) -> None:
        """
        Add an instance to the session for insertion.

        Args:
            instance: Model instance to add
        """
        self._new.add(instance)

        # || Only add nodes to identity map, relationships don't have primary keys
        model_class = type(instance)
        if hasattr(model_class, '__kuzu_node_name__'):
            # This is a node, try to get primary key
            pk_value = self._get_primary_key(instance)
            if pk_value is not None:
                identity_key = self._generate_identity_key(model_class, pk_value)
                self._identity_map[identity_key] = instance

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
        Perform fast bulk insert using PyArrow and Kuzu's COPY FROM.

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
            # Always use bulk insert; multi-pair relationships are handled via
            # pair-qualified COPY in _process_batch_with_pyarrow().
            self._bulk_insert_model_type(model_class, model_instances, effective_batch_size)

    def _bulk_insert_model_type(self, model_class: Type[Any], instances: List[Any], batch_size: int) -> None:
        """
        Bulk insert instances of a single model type using PyArrow with dynamic batch sizing.

        Implements adaptive halving on buffer exhaustion errors to respect Kuzu's buffer pool
        capacity without hard-failing the entire operation. When the batch size reaches 1 and
        still fails, the method escalates the error to the caller.

        Args:
            model_class: The model class type
            instances: List of instances of the same model type
            batch_size: Initial number of records per batch (upper bound)
        """
        if not instances:
            return

        start = 0
        n = len(instances)
        # Clamp batch size to positive
        cur_batch = max(1, int(batch_size))

        def is_buffer_exhaustion_error(err: BaseException) -> bool:
            msg = str(err).lower()
            return (
                "buffer manager exception" in msg
                or "no more frame groups" in msg
                or "out of memory" in msg
            )

        while start < n:
            # Fit the current batch within remaining items
            cur_batch = min(cur_batch, n - start)
            try:
                self._process_batch_with_pyarrow(model_class, instances[start:start + cur_batch])
                start += cur_batch
                # Attempt to gradually increase batch size back up (gentle ramp-up)
                if cur_batch < batch_size:
                    cur_batch = min(batch_size, int(cur_batch * 2))
            except (RuntimeError, ValueError) as e:
                if not is_buffer_exhaustion_error(e):
                    # Non-buffer error: bubble up immediately
                    raise
                # Buffer exhaustion: halve the batch and back off
                if cur_batch <= 1:
                    # Cannot reduce further; re-raise to signal unrecoverable capacity issue
                    raise
                cur_batch = max(1, cur_batch // 2)
                time.sleep(0.02)
                if self._force_gc:
                    gc.collect()
                # Do not advance 'start'; retry with smaller batch

    def _process_batch_with_pyarrow(self, model_class: Type[Any], instances: List[Any]) -> None:
        """
        Process a batch of instances using PyArrow table with explicit UUID schema.
        """
        if not instances:
            return

        # Convert instances to dictionary format
        data_dict = {}
        schema_fields = []

        from .kuzu_orm import KuzuRelationshipBase
        is_relationship = issubclass(model_class, KuzuRelationshipBase)

        if is_relationship:
            # Resolve relationship pairs
            pairs = getattr(model_class, '__kuzu_relationship_pairs__', [])

            # If multi-pair, group instances by (FROM label, TO label) and COPY per pair
            if pairs and len(pairs) > 1:
                # Determine property fields (exclude internal fields)
                all_field_names = list(model_class.model_fields.keys())
                internal_fields = {
                    DDLConstants.REL_FROM_NODE_FIELD,
                    DDLConstants.REL_TO_NODE_FIELD,
                    DDLConstants.REL_FROM_NODE_PK_FIELD,
                    DDLConstants.REL_TO_NODE_PK_FIELD,
                }
                field_names = [f for f in all_field_names if f not in internal_fields]

                # Cached metadata for type conversions
                auto_increment_metadata = model_class.get_auto_increment_metadata()

                # Helper to resolve node label from node instance/reference
                def _resolve_node_label(node_obj: Any) -> str:
                    # Prefer explicit kuzu node name on class or instance
                    if hasattr(node_obj, '__kuzu_node_name__'):
                        return getattr(node_obj, '__kuzu_node_name__')
                    cls = getattr(node_obj, '__class__', None)
                    if cls is not None and hasattr(cls, '__kuzu_node_name__'):
                        return getattr(cls, '__kuzu_node_name__')
                    # Fallback to class name if available; otherwise this is an error
                    if cls is not None and hasattr(cls, '__name__'):
                        return cls.__name__
                    raise ValueError(
                        "Multi-pair relationship bulk insert requires concrete node instances for from_node/to_node to resolve labels."
                    )

                # Group by (from_label, to_label)
                grouped: Dict[tuple[str, str], List[Any]] = {}
                for inst in instances:
                    try:
                        from_label = _resolve_node_label(inst.from_node)
                        to_label = _resolve_node_label(inst.to_node)
                    except Exception as e:
                        raise ValueError(
                            f"Failed to resolve endpoint labels for {model_class.__name__} instance during bulk insert: {e}"
                        ) from e
                    grouped.setdefault((from_label, to_label), []).append(inst)

                # Determine table name for relationship
                if hasattr(model_class, '__kuzu_rel_name__'):
                    table_name = model_class.__kuzu_rel_name__
                elif hasattr(model_class, '__kuzu_relationship_name__'):
                    table_name = model_class.__kuzu_relationship_name__
                else:
                    raise ValueError(
                        f"Model {model_class.__name__} is not a registered relationship"
                    )

                # Execute COPY per (FROM, TO) group
                for (from_label, to_label), group_instances in grouped.items():
                    # Build per-group data dict and schema
                    g_data: Dict[str, List[Any]] = {
                        'from_node_pk': [],
                        'to_node_pk': [],
                    }
                    g_schema: List[pa.Field] = [
                        pa.field('from_node_pk', pa.string()),
                        pa.field('to_node_pk', pa.string()),
                    ]

                    # Initialize property columns
                    for fname in field_names:
                        g_data[fname] = []
                        fmeta = auto_increment_metadata.get(fname)
                        if fmeta and fmeta.kuzu_type == KuzuDataType.UUID:
                            g_schema.append(pa.field(fname, pa.string()))

                    # Fill columns (column-first to minimize Python overhead)
                    # Primary keys as strings for Arrow string schema
                    g_data['from_node_pk'] = [
                        (str(inst.from_node_pk) if inst.from_node_pk is not None else None)
                        for inst in group_instances
                    ]
                    g_data['to_node_pk'] = [
                        (str(inst.to_node_pk) if inst.to_node_pk is not None else None)
                        for inst in group_instances
                    ]

                    # Precompute simple per-field converters based on metadata
                    def _conv_uuid(v: Any) -> Any:
                        return None if v is None else (str(v) if isinstance(v, uuid.UUID) else v)

                    def _conv_default(v: Any) -> Any:
                        # Datetime-like -> ISO; Enums -> value; UUID -> str; list-of-UUID/Enums -> mapped; else pass-through
                        if v is None:
                            return None
                        if isinstance(v, uuid.UUID):
                            return str(v)
                        if isinstance(v, list) and v:
                            first = v[0]
                            if isinstance(first, uuid.UUID):
                                return [str(u) for u in v]
                            from enum import Enum as _E
                            if isinstance(first, _E):
                                return [getattr(u, 'value', u) for u in v]
                        from enum import Enum as _E
                        if isinstance(v, _E):
                            return getattr(v, 'value', v)
                        if hasattr(v, 'isoformat'):
                            try:
                                return v.isoformat()
                            except Exception:
                                return str(v)
                        return v

                    for fname in field_names:
                        fmeta = auto_increment_metadata.get(fname)
                        vals = [getattr(inst, fname, None) for inst in group_instances]
                        if fmeta and fmeta.kuzu_type == KuzuDataType.UUID:
                            g_data[fname] = [_conv_uuid(v) for v in vals]
                        else:
                            # Preserve semantics: coerce most values toward strings later
                            g_data[fname] = [_conv_default(v) for v in vals]

                    # Build PyArrow table
                    arrays = []
                    names = []
                    schema_names = {f.name for f in g_schema}

                    # from_node_pk/to_node_pk remain strings
                    for f in g_schema:
                        arrays.append(pa.array(g_data[f.name], type=f.type))
                        names.append(f.name)

                    # For property fields without explicit schema, let Arrow infer native types
                    # (we already normalized UUIDs/enums/datetimes in g_data).
                    for fname in field_names:
                        if fname not in schema_names:
                            arrays.append(pa.array(g_data[fname]))
                            names.append(fname)

                    g_df = pa.table(arrays, names=names)

                    try:
                        self._execute_with_connection_reuse(
                            f"COPY {table_name} FROM $dataframe (FROM='{from_label}', TO='{to_label}')",
                            {"dataframe": g_df},
                        )
                    finally:
                        try:
                            del g_df
                        except Exception:
                            pass
                        if self._force_gc:
                            gc.collect()

                # Multi-pair relationships handled; no node auto-increment retrieval
                return

            # Handle relationships
            all_field_names = list(model_class.model_fields.keys())
            internal_fields = {
                DDLConstants.REL_FROM_NODE_FIELD,
                DDLConstants.REL_TO_NODE_FIELD,
                DDLConstants.REL_FROM_NODE_PK_FIELD,
                DDLConstants.REL_TO_NODE_PK_FIELD,
            }
            field_names = [f for f in all_field_names if f not in internal_fields]

            # Add node reference columns
            data_dict['from_node_pk'] = []
            data_dict['to_node_pk'] = []
            schema_fields.extend([
                pa.field('from_node_pk', pa.string()),  # Use string for UUID compatibility
                pa.field('to_node_pk', pa.string()),    # Use string for UUID compatibility
            ])

            # Initialize property columns with proper types
            auto_increment_metadata = model_class.get_auto_increment_metadata()
            for field_name in field_names:
                data_dict[field_name] = []

                # Determine PyArrow type for this field
                field_meta = auto_increment_metadata.get(field_name)
                if field_meta and field_meta.kuzu_type == KuzuDataType.UUID:
                    # Use string type for UUID fields to avoid BLOB conversion issues
                    schema_fields.append(pa.field(field_name, pa.string()))

            # Extract columns from relationship instances (column-first for speed)
            data_dict['from_node_pk'].extend([
                (str(inst.from_node_pk) if inst.from_node_pk is not None else None)
                for inst in instances
            ])
            data_dict['to_node_pk'].extend([
                (str(inst.to_node_pk) if inst.to_node_pk is not None else None)
                for inst in instances
            ])

            from enum import Enum as _E
            for field_name in field_names:
                fmeta = auto_increment_metadata.get(field_name)
                col = []
                append = col.append
                for inst in instances:
                    value = getattr(inst, field_name, None)
                    if value is None:
                        append(None)
                        continue
                    if fmeta and fmeta.kuzu_type == KuzuDataType.UUID and isinstance(value, uuid.UUID):
                        append(str(value))
                        continue
                    if isinstance(value, uuid.UUID):
                        append(str(value))
                        continue
                    if isinstance(value, _E):
                        append(getattr(value, 'value', value))
                        continue
                    if hasattr(value, 'isoformat'):
                        try:
                            append(value.isoformat())
                        except Exception:
                            append(str(value))
                        continue
                    if (hasattr(value, '__class__') and 'KuzuDefaultFunction' in str(value.__class__)):
                        append(self._generate_default_function_value(value))
                        continue
                    append(value)
                data_dict[field_name].extend(col)

        else:
            # Handle nodes
            all_field_names = list(model_class.model_fields.keys())
            auto_increment_fields = model_class.get_auto_increment_fields()

            # Determine which fields to include
            sample_instance = instances[0]
            sample_data = sample_instance.model_dump()

            field_names = []
            for field_name in all_field_names:
                if field_name in auto_increment_fields and sample_data.get(field_name) is None:
                    continue
                field_names.append(field_name)

            # Initialize columns with proper schema
            auto_increment_metadata = model_class.get_auto_increment_metadata()
            for field_name in field_names:
                data_dict[field_name] = []

                # Determine PyArrow type for this field
                field_meta = auto_increment_metadata.get(field_name)
                if field_meta and field_meta.kuzu_type == KuzuDataType.UUID:
                    # Use string type for UUID fields to avoid BLOB conversion issues
                    schema_fields.append(pa.field(field_name, pa.string()))
                # Note: For other types, we'll let PyArrow infer from the data later

            # Extract data from instances (column-first for speed)
            from enum import Enum as _E
            for field_name in field_names:
                fmeta = auto_increment_metadata.get(field_name)
                col = []
                append = col.append
                for inst in instances:
                    value = getattr(inst, field_name, None)
                    if value is None:
                        append(None)
                        continue
                    if fmeta and fmeta.kuzu_type == KuzuDataType.UUID and isinstance(value, uuid.UUID):
                        append(str(value))
                        continue
                    if isinstance(value, uuid.UUID):
                        append(str(value))
                        continue
                    if isinstance(value, _E):
                        append(getattr(value, 'value', value))
                        continue
                    if hasattr(value, 'isoformat'):
                        try:
                            append(value.isoformat())
                        except Exception:
                            append(str(value))
                        continue
                    if (hasattr(value, '__class__') and 'KuzuDefaultFunction' in str(value.__class__)):
                        append(self._generate_default_function_value(value))
                        continue
                    append(value)
                data_dict[field_name].extend(col)

        # Create PyArrow table with mixed explicit and inferred schema
        if schema_fields:
            # Build arrays with proper types for explicit fields, let PyArrow infer others
            arrays = []
            field_names_ordered = []
            schema_field_names = {f.name for f in schema_fields}

            # First, add arrays for fields with explicit types
            for field in schema_fields:
                arrays.append(pa.array(data_dict[field.name], type=field.type))
                field_names_ordered.append(field.name)

            # Then, add arrays for fields without explicit types (let PyArrow infer)
            for field_name in field_names:
                if field_name not in schema_field_names:
                    # Convert any remaining UUID objects to strings before PyArrow inference
                    field_data = data_dict[field_name]
                    converted_data = []
                    for value in field_data:
                        if isinstance(value, uuid.UUID):
                            converted_data.append(str(value))
                        elif isinstance(value, list) and value and isinstance(value[0], uuid.UUID):
                            converted_data.append([str(uuid_val) for uuid_val in value])
                        else:
                            converted_data.append(value)
                    arrays.append(pa.array(converted_data))
                    field_names_ordered.append(field_name)

            # Create table with field names only, let PyArrow infer types for non-explicit fields
            df = pa.table(arrays, names=field_names_ordered)
        else:
            # Fallback to full inference - convert all UUID objects to strings first
            converted_dict = {}
            for field_name, field_data in data_dict.items():
                converted_data = []
                for value in field_data:
                    if isinstance(value, uuid.UUID):
                        converted_data.append(str(value))
                    elif isinstance(value, list) and value and isinstance(value[0], uuid.UUID):
                        converted_data.append([str(uuid_val) for uuid_val in value])
                    else:
                        converted_data.append(value)
                converted_dict[field_name] = converted_data
            df = pa.table(converted_dict)

        data_dict.clear()
        del data_dict

        # Determine table name and execute COPY
        if hasattr(model_class, '__kuzu_node_name__'):
            table_name = model_class.__kuzu_node_name__
        elif hasattr(model_class, '__kuzu_rel_name__'):
            table_name = model_class.__kuzu_rel_name__
        elif hasattr(model_class, '__kuzu_relationship_name__'):
            table_name = model_class.__kuzu_relationship_name__
        else:
            raise ValueError(f"Model {model_class.__name__} is not a registered node or relationship")

        try:
            self._execute_with_connection_reuse(f"COPY {table_name} FROM $dataframe", {"dataframe": df})
        finally:
            # Ensure PyArrow buffers are released promptly
            try:
                del df
            except Exception:
                pass
            if self._force_gc:
                gc.collect()

        if not is_relationship:
            self._retrieve_auto_increment_values_after_bulk_insert(model_class, instances, field_names)

    def _retrieve_auto_increment_values_after_bulk_insert(self, model_class: Type[Any], instances: List[Any], included_field_names: List[str]) -> None:
        """
        Retrieve auto-increment values after bulk insert and update instances.

        Since COPY FROM doesn't return generated auto-increment values like CREATE does,
        we need to query the database to retrieve the generated values and update the instances.

        Args:
            model_class: The model class type
            instances: List of instances that were bulk inserted
            included_field_names: Field names that were included in the bulk insert DataFrame
        """
        # @@ STEP 1: Get auto-increment fields that need retrieval
        auto_increment_fields = model_class.get_auto_increment_fields()
        if not auto_increment_fields:
            return  # No auto-increment fields to retrieve

        # @@ STEP 2: Determine which auto-increment fields were excluded from bulk insert
        excluded_auto_increment_fields = [f for f in auto_increment_fields if f not in included_field_names]
        if not excluded_auto_increment_fields:
            return  # All auto-increment fields were included (had non-None values)

        # @@ STEP 3: Build query to retrieve generated values
        # || S.S.1: We need to match inserted records by their non-auto-increment fields
        try:
            table_name = model_class.__kuzu_node_name__
        except AttributeError as attr_err:
            raise RuntimeError(
                f"Model class {model_class.__name__} does not have __kuzu_node_name__ attribute. "
                f"This indicates a malformed node class definition. "
                f"Original error: {attr_err}"
            ) from attr_err

        # || S.S.2: Get non-auto-increment fields for matching
        matching_fields = [f for f in included_field_names if f not in auto_increment_fields]
        if not matching_fields:
            raise RuntimeError(
                f"Cannot retrieve auto-increment values for {model_class.__name__}: "
                f"no non-auto-increment fields available for matching. "
                f"At least one non-auto-increment field is required to identify inserted records. "
                f"Included fields: {included_field_names}, Auto-increment fields: {auto_increment_fields}"
            )

        # @@ STEP 4: For each instance, query to get the generated auto-increment values
        for instance in instances:
            try:
                # || S.S.3: Build WHERE conditions using non-auto-increment fields
                where_conditions = []
                params = {}

                for field_name in matching_fields:
                    field_value = getattr(instance, field_name)
                    param_name = f"match_{field_name}"
                    where_conditions.append(f"n.{field_name} = ${param_name}")
                    params[param_name] = field_value

                # || S.S.4: Build query to retrieve auto-increment values
                select_fields = ", ".join([f"n.{field}" for field in excluded_auto_increment_fields])
                where_clause = " AND ".join(where_conditions)

                query = f"""
                MATCH (n:{table_name})
                WHERE {where_clause}
                RETURN {select_fields}
                LIMIT 1
                """

                # || S.S.5: Execute query and update instance
                result = self._execute_with_connection_reuse(query, params)

                if result and len(result) > 0:
                    record = result[0]
                    for field_name in excluded_auto_increment_fields:
                        # || The query returns 'n.field_name' but we need just 'field_name'
                        record_key = f"n.{field_name}"
                        generated_value = record.get(record_key)
                        if generated_value is not None:
                            setattr(instance, field_name, generated_value)
                else:
                    # || S.S.6: Log warning if no matching record found
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Could not retrieve auto-increment values for {model_class.__name__} instance. "
                        f"No matching record found with conditions: {where_conditions}"
                    )

            except Exception as e:
                # || S.S.7: Log error but don't fail the entire operation
                logger = logging.getLogger(__name__)
                logger.error(
                    f"Failed to retrieve auto-increment values for {model_class.__name__} instance: {e}. "
                    f"Auto-increment fields will remain None: {excluded_auto_increment_fields}"
                )

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

        # @@ STEP 1: Only handle nodes for merge, relationships don't have primary keys
        if not hasattr(model_class, '__kuzu_node_name__'):
            # This is a relationship, just add it
            self.add(instance)
            return instance

        # Get primary key value for nodes
        pk_value = self._get_primary_key(instance)
        if not pk_value:
            self.add(instance)
            return instance

        identity_key = self._generate_identity_key(model_class, pk_value)
        existing = self._identity_map.get(identity_key)

        if existing is not None:
            # Update existing with new values
            for field_name, field_value in instance.model_dump().items():
                setattr(existing, field_name, field_value)
            self._dirty.add(existing)
            return existing
        else:
            # Add new instance to _new, not _dirty. Non-existent instances should be treated as new insertions
            self._identity_map[identity_key] = instance
            self._new.add(instance)
            return instance

    def flush(self) -> None:
        """
        Flush pending changes to the database without committing.
        """
        if self._flushing:
            return

        self._flushing = True
        try:
            # Reset connection reuse for batch operations
            self._reset_connection_reuse()

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
        # Only expire nodes, relationships don't have identity map entries
        model_class = type(instance)
        if hasattr(model_class, '__kuzu_node_name__'):
            pk_value = self._get_primary_key(instance)
            if pk_value is not None:
                # Use optimized identity key generation
                identity_key = self._generate_identity_key(model_class, pk_value)
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

        # || S.1: Check cache first for O(1) lookup
        cached_pk_fields = self._get_cached_metadata(model_class, 'pk_fields')

        if cached_pk_fields is not None:
            pk_fields = cached_pk_fields
        elif hasattr(model_class, 'get_primary_key_fields'):
            get_pk_fields = model_class.get_primary_key_fields
            if not callable(get_pk_fields):
                raise ValueError(
                    f"Model {model_class.__name__}.get_primary_key_fields exists but is not callable"
                )
            pk_fields = cast(List[str], get_pk_fields())
            # || S.2: Cache the result for future use
            self._set_cached_metadata(model_class, 'pk_fields', pk_fields)
        else:
            pk_fields = None

        if pk_fields:

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

                # @@ STEP 3: Validate that we got a non-None value (unless it's auto-increment)
                if value is None:
                    # || S.S.1: All KuzuBaseModel instances have get_auto_increment_fields() method
                    # || S.S.2: No fallbacks - direct method call with explicit error handling
                    try:
                        auto_increment_fields = model_class.get_auto_increment_fields()
                        if pk_field in auto_increment_fields:
                            # || S.S.3: Auto-increment primary keys can be None (will be generated by DB)
                            # || Unset auto-increment fields have no value until DB generation
                            return None
                    except AttributeError as attr_err:
                        # || S.S.4: Explicit error for non-KuzuBaseModel classes
                        raise TypeError(
                            f"Model class {model_class.__name__} does not implement get_auto_increment_fields() method. "
                            f"All model classes must inherit from KuzuBaseModel. "
                            f"Original error: {attr_err}"
                        ) from attr_err

                    # || S.S.5: Explicit error with detailed context for non-auto-increment fields
                    raise ValueError(
                        f"Primary key field '{pk_field}' is None on {model_class.__name__} instance. "
                        f"The field exists but has no value. Non-auto-increment primary keys must have a non-None value."
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

        # @@ STEP 1: Analyze auto-increment field requirements
        # || S.1.1: Get fields that need database-generated values (not explicitly set)
        fields_needing_generation = instance.get_auto_increment_fields_needing_generation()
        # || S.1.2: Get fields that were manually provided during instantiation
        manual_auto_increment_values = instance.get_manual_auto_increment_values()

        # Validate manual auto-increment values
        self._validate_manual_auto_increment_values(manual_auto_increment_values, model_class)

        # Build CREATE query - include manual auto-increment values in properties
        properties = instance.model_dump(exclude_unset=True)

        # @@ STEP 2: Add manual auto-increment values with strict type validation
        auto_increment_metadata = model_class.get_auto_increment_metadata()

        for field_name, value in manual_auto_increment_values.items():
            if value is not None:  # Only include non-None manual values
                # || S.2.1: Validate UUID fields must be UUID objects, not strings
                field_meta = auto_increment_metadata.get(field_name)
                if field_meta and field_meta.kuzu_type == KuzuDataType.UUID:
                    if isinstance(value, str):
                        # || S.2.1.2: Reject strings - enforce UUID objects only
                        raise TypeError(
                            ErrorMessages.UUID_STRING_NOT_ALLOWED.format(
                                field_name=field_name,
                                model_name=model_class.__name__,
                                value=value,
                                type_name=type(value).__name__
                            )
                        )
                    elif not isinstance(value, uuid.UUID):
                        # || S.2.1.3: Reject any non-UUID object types
                        raise TypeError(
                            f"UUID field '{field_name}' in {model_class.__name__} "
                            f"must be a UUID object, got: {value} ({type(value).__name__})"
                        )
                    # || S.2.1.4: Value is already a proper UUID object
                    converted_value = value
                else:
                    # || S.2.1.5: For SERIAL fields, use value as-is
                    converted_value = value

                properties[field_name] = converted_value

        # @@ STEP 2.5: Normalize property values for prepared statement binding
        # Convert Enums to their underlying values, UUIDs to strings, and lists of UUIDs to list[str]
        all_meta = model_class.get_all_kuzu_metadata()

        normalized_props: Dict[str, Any] = {}
        for k, v in properties.items():
            meta = all_meta.get(k)
            # Enum -> underlying value
            if isinstance(v, Enum):
                v = getattr(v, 'value', v)
            # UUID -> string
            if isinstance(v, uuid.UUID):
                v = str(v)
            # datetime-like -> isoformat
            if hasattr(v, 'isoformat'):
                v = v.isoformat()
            # Handle ARRAY(UUID) -> list[str]
            if meta is not None:
                from .kuzu_orm import ArrayTypeSpecification, KuzuDataType as _KDT
                ktype = meta.kuzu_type
                if isinstance(ktype, ArrayTypeSpecification):
                    elem_t = ktype.element_type
                    is_uuid_elem = (elem_t == _KDT.UUID) or (isinstance(elem_t, str) and str(elem_t).upper() == 'UUID')
                    if is_uuid_elem and isinstance(v, list):
                        v = [str(x) if isinstance(x, uuid.UUID) else x for x in v]
            normalized_props[k] = v
        properties = normalized_props

        if not properties:
            query = f"CREATE (:{node_name})"
            self._execute_with_connection_reuse(query)
        else:
            prop_str = ", ".join(f"{k}: ${k}" for k in properties.keys())
            query = f"CREATE (:{node_name} {{{prop_str}}})"
            self._execute_with_connection_reuse(query, properties)

        # Handle auto-increment fields that need generation: retrieve generated values and update instance
        if fields_needing_generation:
            self._handle_auto_increment_after_insert(instance, model_class, node_name, fields_needing_generation)

        # Add to identity map with optimized key generation
        pk_value = self._get_primary_key(instance)
        if pk_value is not None:
            identity_key = self._generate_identity_key(model_class, pk_value)
            self._identity_map[identity_key] = instance

    def _handle_auto_increment_after_insert(self, instance: Any, model_class: Type[Any], node_name: str, auto_increment_fields: List[str]) -> None:
        """
        Handle auto-increment fields after node insertion.

        Retrieves the auto-generated values from the database and updates the instance.
        This is necessary because KuzuDB generates SERIAL values automatically during INSERT,
        but we need to retrieve them to update the Python instance.

        Args:
            instance: The model instance that was just inserted
            model_class: The model class type
            node_name: The node table name in KuzuDB
            auto_increment_fields: List of field names that are auto-increment
        """
        # Strategy: Query the most recently inserted node to get auto-generated values
        # We'll use the non-auto-increment fields to identify the specific node we just inserted
        non_auto_fields = {}
        for field_name, value in instance.model_dump(exclude_unset=True).items():
            if field_name not in auto_increment_fields and value is not None:
                non_auto_fields[field_name] = value

        try:
            if non_auto_fields:
                # Build WHERE clause using non-auto-increment fields
                where_conditions = []
                params = {}
                for field_name, value in non_auto_fields.items():
                    where_conditions.append(f"n.{field_name} = ${field_name}")
                    params[field_name] = value

                where_clause = " AND ".join(where_conditions)

                # Order by the first auto-increment field (SERIAL fields are sequential)
                order_field = auto_increment_fields[0]
                query = f"MATCH (n:{node_name}) WHERE {where_clause} RETURN n ORDER BY n.{order_field} DESC LIMIT 1"

                result = self._execute_with_connection_reuse(query, params)
            else:
                # If no non-auto fields, get the most recent node by the first auto-increment field
                order_field = auto_increment_fields[0]
                query = f"MATCH (n:{node_name}) RETURN n ORDER BY n.{order_field} DESC LIMIT 1"
                result = self._execute_with_connection_reuse(query)

        except Exception as e:
            # Re-raise with context about the failed operation
            raise RuntimeError(
                f"Failed to retrieve auto-generated values for {model_class.__name__} after successful insertion. "
                f"This indicates a serious database connectivity or schema issue. "
                f"Original error: {e}"
            ) from e

        # Handle the case where no results are found
        if not result or len(result) == 0:
            logger.warning(
                f"Could not retrieve auto-generated values for {model_class.__name__} - "
                f"no matching records found. This may indicate a timing issue or concurrent access. "
                f"Auto-increment fields will remain unset: {auto_increment_fields}"
            )
            return

        # Extract and update auto-generated values
        node_data = result[0]['n']
        updated_fields = []

        # @@ STEP 3: Get auto-increment metadata for proper type conversion
        auto_increment_metadata = model_class.get_auto_increment_metadata()

        for field_name in auto_increment_fields:
            if field_name in node_data:
                raw_value = node_data[field_name]

                # || S.3.1: Convert UUID objects to strings for UUID fields
                field_meta = auto_increment_metadata.get(field_name)
                if field_meta and field_meta.kuzu_type == KuzuDataType.UUID:
                    # || S.3.1.1: KuzuDB returns UUID objects, convert to string
                    converted_value = str(raw_value)
                else:
                    # || S.3.1.2: For SERIAL fields, use value as-is
                    converted_value = raw_value

                setattr(instance, field_name, converted_value)
                updated_fields.append(f"{field_name}={converted_value}")

        if updated_fields:
            logger.debug(f"Retrieved auto-generated values for {model_class.__name__}: {', '.join(updated_fields)}")
        else:
            logger.warning(
                f"No auto-increment field values found in retrieved data for {model_class.__name__}. "
                f"Expected fields: {auto_increment_fields}, Retrieved data keys: {list(node_data.keys())}"
            )

    def _determine_relationship_pair(self, instance: Any, model_class: Type[Any]) -> Any:
        """
        Determine the correct RelationshipPair for a relationship instance.

        Determines which FROM-TO pair matches the actual node types
        in the relationship instance. This ensures multi-pair relationships work correctly.

        Args:
            instance: The relationship instance with from_node and to_node
            model_class: The relationship model class

        Returns:
            The matching RelationshipPair

        Raises:
            ValueError: If no matching pair found or multiple ambiguous matches
            RuntimeError: If relationship pairs are malformed
        """
        # @@ STEP 1: Get all relationship pairs
        try:
            rel_pairs = instance.get_relationship_pairs()
        except AttributeError as attr_err:
            raise RuntimeError(
                f"Relationship instance {model_class.__name__} does not implement get_relationship_pairs() method. "
                f"This indicates a malformed relationship class. "
                f"Original error: {attr_err}"
            ) from attr_err

        if not rel_pairs:
            raise ValueError(f"No relationship pairs defined for {model_class.__name__}")

        # @@ STEP 2: Determine actual node types from the instance
        from_node = instance.from_node
        to_node = instance.to_node

        if from_node is None or to_node is None:
            raise ValueError(
                f"Relationship {model_class.__name__} must have both from_node and to_node specified. "
                f"Got from_node={from_node}, to_node={to_node}"
            )

        # @@ STEP 3: Get node type names
        from_node_type_name = self._get_node_type_name(from_node)
        to_node_type_name = self._get_node_type_name(to_node)

        # @@ STEP 4: Find matching pair using set membership
        matching_pairs = []
        for pair in rel_pairs:
            pair_from_name = pair.get_from_name()
            pair_to_name = pair.get_to_name()

            if pair_from_name == from_node_type_name and pair_to_name == to_node_type_name:
                matching_pairs.append(pair)

        # @@ STEP 5: Validate exactly one match (uniqueness constraint)
        if len(matching_pairs) == 0:
            available_pairs = [f"({p.get_from_name()}, {p.get_to_name()})" for p in rel_pairs]
            raise ValueError(
                f"No matching relationship pair found for {model_class.__name__} "
                f"with from_node type '{from_node_type_name}' and to_node type '{to_node_type_name}'. "
                f"Available pairs: {available_pairs}"
            )
        elif len(matching_pairs) > 1:
            # This should not happen with proper relationship definitions
            duplicate_pairs = [f"({p.get_from_name()}, {p.get_to_name()})" for p in matching_pairs]
            raise ValueError(
                f"Multiple matching relationship pairs found for {model_class.__name__} "
                f"with from_node type '{from_node_type_name}' and to_node type '{to_node_type_name}'. "
                f"Duplicate pairs: {duplicate_pairs}. This indicates a malformed relationship definition."
            )

        return matching_pairs[0]

    def _get_node_type_name(self, node: Any) -> str:
        """
        Get the node type name from a node instance or raw primary key value.

        Implements complete node type determination without fallbacks.

        Args:
            node: Node instance or primary key value

        Returns:
            The node type name

        Raises:
            TypeError: If node type cannot be determined due to invalid input
            RuntimeError: If registry lookup fails
        """
        # @@ STEP 1: Handle model instances (direct case)
        if hasattr(node, '__class__') and hasattr(node.__class__, '__kuzu_node_name__'):
            return node.__class__.__kuzu_node_name__

        # @@ STEP 2: Handle raw primary key values by querying the database
        # || S.S.1: For raw PK values, we need to determine which node type has this PK
        # || S.S.2: Query all registered node types to find the matching one
        registered_nodes = get_registered_nodes()
        if not registered_nodes:
            raise RuntimeError(
                f"No registered node types found in registry. "
                f"Cannot determine node type for primary key value: {node}"
            )

        # @@ STEP 3: Query each node type to find which one contains this primary key
        matching_node_types = []

        for node_name, node_class in registered_nodes.items():
            try:
                # || S.S.3: Get the primary key field name for this node type
                pk_field = self._get_primary_key_field_name(node_class)

                # || S.S.4: Query to check if this node type contains the primary key value
                query = f"MATCH (n:{node_name}) WHERE n.{pk_field} = $pk_value RETURN count(n) as count"
                result = self._execute_with_connection_reuse(query, {'pk_value': node})

                if result and len(result) > 0 and result[0].get('count', 0) > 0:
                    matching_node_types.append(node_name)

            # Replace bare exception handling with specific exception types
            except ValueError as e:
                # || S.S.5a: Model class has no primary key field - log and continue
                logger.debug(
                    ValidationMessageConstants.NO_PRIMARY_KEY_FIELD.format(node_class.__name__) +
                    f" - {str(e)}"
                )
                continue
            except (RuntimeError, ConnectionError, TimeoutError) as e:
                # || S.S.5b: Database connection or execution errors - log and continue
                logger.debug(
                    ValidationMessageConstants.DATABASE_QUERY_FAILED.format(node_name, node, str(e))
                )
                continue
            except AttributeError as e:
                # || S.S.5c: Model class missing required attributes - log and continue
                logger.debug(
                    ValidationMessageConstants.MODEL_ATTRIBUTE_ERROR.format(node_class.__name__, str(e))
                )
                continue
            except (TypeError, KeyError) as e:
                # || S.S.5d: Parameter binding or result parsing errors - log and continue
                if "parameter" in str(e).lower() or "bind" in str(e).lower():
                    logger.debug(
                        ValidationMessageConstants.PARAMETER_BINDING_ERROR.format(node_name, node, str(e))
                    )
                else:
                    logger.debug(
                        ValidationMessageConstants.RESULT_PARSING_ERROR.format(node_name, str(e))
                    )
                continue
            except Exception as e:
                # || S.S.5e: Unexpected errors - log with full context and continue
                logger.warning(
                    f"Unexpected error while checking node type '{node_name}' for primary key value '{node}': "
                    f"{type(e).__name__}: {str(e)}. Continuing with other node types."
                )
                continue

        # @@ STEP 4: Handle multiple matches (normal in graph databases)
        if len(matching_node_types) == 0:
            available_types = list(registered_nodes.keys())
            raise TypeError(
                f"Primary key value {node} does not exist in any registered node type. "
                f"Available node types: {available_types}. "
                f"Ensure the node exists in the database and the primary key value is correct."
            )
        elif len(matching_node_types) > 1:
            # || In graph databases, primary keys are unique within node types, not globally
            # || Multiple node types can have the same primary key value
            # || This is normal and expected behavior
            raise TypeError(
                f"Primary key value {node} exists in multiple node types: {matching_node_types}. "
                f"Cannot determine unique node type from primary key alone in graph database. "
                f"Use model instances instead of raw primary key values for multi-pair relationships, "
                f"or provide additional context to disambiguate the node type."
            )

        return matching_node_types[0]

    def _validate_manual_auto_increment_values(self, manual_values: Dict[str, Any], model_class: Type[Any]) -> None:
        """
        Validate manually provided auto-increment values.

        Args:
            manual_values: Dictionary of field names to manually provided values
            model_class: The model class being validated

        Raises:
            ValueError: If any manual values are invalid
            TypeError: If input parameters are invalid
        """
        # @@ STEP 1: Defensive validation of input parameters
        # || S.1.1: Validate manual_values parameter
        if not isinstance(manual_values, dict):
            raise TypeError(
                f"manual_values must be a dictionary, got: {type(manual_values).__name__}"
            )

        # || S.1.2: Validate model_class parameter
        if not isinstance(model_class, type) or not hasattr(model_class, 'get_auto_increment_metadata'):
            raise TypeError(
                f"model_class must be a KuzuBaseModel class, got: {type(model_class).__name__}"
            )

        # || S.1.3: Early return optimization for empty manual values
        if not manual_values:
            return

        # @@ STEP 2: Get auto-increment field metadata for validation
        try:
            auto_increment_metadata = model_class.get_auto_increment_metadata()
        except Exception as e:
            raise ValueError(
                f"Failed to get auto-increment metadata for {model_class.__name__}: {e}"
            ) from e

        # @@ STEP 3: Validate each manual value with comprehensive error handling
        for field_name, value in manual_values.items():
            # || S.3.1: Validate field name
            if not isinstance(field_name, str):
                raise TypeError(
                    f"Field names must be strings, got: {type(field_name).__name__} for value {value}"
                )

            # || S.3.2: Skip None values (they are handled separately)
            if value is None:
                continue

            # || S.3.3: Get field metadata to determine validation rules
            field_meta = auto_increment_metadata.get(field_name)
            if not field_meta:
                # || S.3.4: Field not found in metadata - this could indicate a bug
                continue  # Skip validation for unknown fields (defensive)

            # || S.3.5: Validate based on field type with comprehensive error messages
            if field_meta.kuzu_type == KuzuDataType.SERIAL:
                # || S.3.5.1: SERIAL fields must be non-negative integers
                if not isinstance(value, int):
                    raise ValueError(
                        f"Auto-increment SERIAL field '{field_name}' in {model_class.__name__} "
                        f"must be an integer, got: {value} ({type(value).__name__}). "
                        f"SERIAL fields only accept non-negative integer values."
                    )
                if value < 0:
                    raise ValueError(
                        f"Auto-increment SERIAL field '{field_name}' in {model_class.__name__} "
                        f"must be non-negative, got: {value}. SERIAL fields start from 0."
                    )
            elif field_meta.kuzu_type == KuzuDataType.UUID:
                # || S.3.5.2: UUID fields must be UUID objects only
                if isinstance(value, str):
                    # || S.3.5.2.1: Reject strings - enforce UUID objects only
                    raise TypeError(
                        ErrorMessages.UUID_STRING_NOT_ALLOWED.format(
                            field_name=field_name,
                            model_name=model_class.__name__,
                            value=value,
                            type_name=type(value).__name__
                        )
                    )
                elif not isinstance(value, uuid.UUID):
                    # || S.3.5.2.2: Reject any non-UUID object types with detailed error
                    raise TypeError(
                        f"UUID field '{field_name}' in {model_class.__name__} "
                        f"must be a UUID object, got: {value} ({type(value).__name__}). "
                        f"Use uuid.UUID() to create proper UUID objects."
                    )

    def _insert_relationship_instance(self, instance: Any) -> None:
        """Insert a relationship instance into the database."""
        model_class = type(instance)
        rel_name = model_class.get_relationship_name()

        # Check if model has auto-increment fields
        fields_needing_generation = instance.get_auto_increment_fields_needing_generation()
        manual_auto_increment_values = instance.get_manual_auto_increment_values()

        # Validate manual auto-increment values
        self._validate_manual_auto_increment_values(manual_auto_increment_values, model_class)

        # Get from and to node primary keys
        from_pk = instance.from_node_pk
        to_pk = instance.to_node_pk

        if from_pk is None or to_pk is None:
            raise ValueError(
                f"Relationship {model_class.__name__} must have both from_node and to_node specified"
            )

        # @@ STEP 1: Determine the correct relationship pair for this instance
        # || S.S.1: Determination of the unique matching pair
        matching_pair = self._determine_relationship_pair(instance, model_class)

        # @@ STEP 2: Get node names from the determined pair
        from_node_name = matching_pair.get_from_name()
        to_node_name = matching_pair.get_to_name()

        # Build relationship properties (exclude internal fields)
        properties = instance.model_dump(exclude_unset=True)
        # Remove internal fields that shouldn't be in the relationship
        internal_fields = {
            DDLConstants.REL_FROM_NODE_FIELD,  # 'from_node'
            DDLConstants.REL_TO_NODE_FIELD,    # 'to_node'
            DDLConstants.REL_FROM_NODE_PK_FIELD,  # Private field for from_node primary key cache
            DDLConstants.REL_TO_NODE_PK_FIELD,    # Private field for to_node primary key cache
        }
        properties = {k: v for k, v in properties.items() if k not in internal_fields}

        # KuzuDB doesn't handle Python enum objects directly, so convert them to their underlying values
        for key, value in properties.items():
            if isinstance(value, Enum):
                properties[key] = value.value

        # Add manual auto-increment values to properties
        for field_name, value in manual_auto_increment_values.items():
            if value is not None:  # Only include non-None manual values
                properties[field_name] = value

        # @@ STEP 3: Determine primary key field names from node names
        # || S.S.1: Use registry resolution for node name to primary key field mapping
        try:
            from_pk_field = self._resolve_pk_field_by_node_name(from_node_name)
            to_pk_field = self._resolve_pk_field_by_node_name(to_node_name)
        except Exception as e:
            # Re-raise with context about the failed operation
            raise RuntimeError(
                f"Failed to determine primary key fields for relationship {model_class.__name__} "
                f"between {from_node_name} and {to_node_name}. "
                f"Original error: {e}"
            ) from e

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

        self._execute_with_connection_reuse(query, params)

        # Handle auto-increment fields that need generation: retrieve generated values and update instance
        if fields_needing_generation:
            self._handle_auto_increment_after_relationship_insert(instance, model_class, rel_name, fields_needing_generation, from_pk, to_pk)

    def _handle_auto_increment_after_relationship_insert(self, instance: Any, model_class: Type[Any], rel_name: str, auto_increment_fields: List[str], from_pk: Any, to_pk: Any) -> None:
        """
        Handle auto-increment fields after relationship insertion.

        Retrieves the auto-generated values from the database and updates the instance.
        This is necessary because KuzuDB generates SERIAL values automatically during CREATE,
        but we need to retrieve them to update the Python relationship instance.

        Args:
            instance: The relationship instance that was just inserted
            model_class: The relationship class type
            rel_name: The relationship name in KuzuDB
            auto_increment_fields: List of field names that are auto-increment
            from_pk: Primary key of the from node
            to_pk: Primary key of the to node
        """
        # Strategy: Query the most recently inserted relationship to get auto-generated values
        # We'll use the from/to node primary keys to identify the specific relationship we just inserted

        # @@ STEP 1: Determine the correct relationship pair for this instance
        # || S.S.1: Determination of the unique matching pair
        matching_pair = self._determine_relationship_pair(instance, model_class)

        # @@ STEP 2: Get node names from the determined pair
        from_node_name = matching_pair.get_from_name()
        to_node_name = matching_pair.get_to_name()

        # Get primary key field names by resolving via registry (no isinstance/hasattr)
        try:
            from_pk_field = self._resolve_pk_field_by_node_name(from_node_name)
            to_pk_field = self._resolve_pk_field_by_node_name(to_node_name)
        except Exception as e:
            # Re-raise with context about the failed operation
            raise RuntimeError(
                f"Failed to determine primary key fields for relationship {model_class.__name__} "
                f"between {from_node_name} and {to_node_name}. "
                f"Original error: {e}"
            ) from e

        # @@ STEP 1: Build query using fluent API components for maintainability
        try:
            # || S.S.1: Order by the first auto-increment field (SERIAL fields are sequential)
            order_field = auto_increment_fields[0]

            # || S.S.2: Use constants and structured query building instead of raw string concatenation
            from .constants import CypherConstants

            # || S.S.3: Build query components using constants for maintainability
            match_pattern = f"(from_node:{from_node_name})-[r:{rel_name}]->(to_node:{to_node_name})"
            where_conditions = [
                f"from_node.{from_pk_field} = $from_pk",
                f"to_node.{to_pk_field} = $to_pk"
            ]

            # || S.S.4: Construct query using constants instead of magic strings
            query_parts = [
                f"{CypherConstants.MATCH} {match_pattern}",
                f"{CypherConstants.WHERE} {f' {CypherConstants.AND} '.join(where_conditions)}",
                f"{CypherConstants.RETURN} r",
                f"{CypherConstants.ORDER_BY} r.{order_field} {CypherConstants.DESC}",
                f"{CypherConstants.LIMIT} 1"
            ]

            query = "\n".join(query_parts)
            params = {'from_pk': from_pk, 'to_pk': to_pk}

            result = self._execute_with_connection_reuse(query, params)
        except Exception as e:
            # Re-raise with context about the failed operation
            raise RuntimeError(
                f"Failed to retrieve auto-generated values for relationship {model_class.__name__} after successful insertion. "
                f"This indicates a serious database connectivity or schema issue. "
                f"Query: {query}, Params: {params}. "
                f"Original error: {e}"
            ) from e

        # Handle the case where no results are found
        if not result or len(result) == 0:
            logger.warning(
                f"Could not retrieve auto-generated values for relationship {model_class.__name__} - "
                f"no matching records found between {from_node_name}(pk={from_pk}) and {to_node_name}(pk={to_pk}). "
                f"This may indicate a timing issue or concurrent access. "
                f"Auto-increment fields will remain unset: {auto_increment_fields}"
            )
            return

        # Extract and update auto-generated values
        rel_data = result[0]['r']
        updated_fields = []

        for field_name in auto_increment_fields:
            if field_name in rel_data:
                setattr(instance, field_name, rel_data[field_name])
                updated_fields.append(f"{field_name}={rel_data[field_name]}")

        if updated_fields:
            logger.debug(f"Retrieved auto-generated values for relationship {model_class.__name__}: {', '.join(updated_fields)}")
        else:
            logger.warning(
                f"No auto-increment field values found in retrieved relationship data for {model_class.__name__}. "
                f"Expected fields: {auto_increment_fields}, Retrieved data keys: {list(rel_data.keys())}"
            )

    def _get_primary_key_field_name(self, model_class: Type[Any]) -> str:
        """Get the primary key field name for a model class."""
        from .kuzu_orm import _kuzu_registry

        for field_name, field_info in model_class.model_fields.items():
            metadata = _kuzu_registry.get_field_metadata(field_info)
            if metadata and metadata.primary_key:
                return field_name
        raise ValueError(f"No primary key field found in {model_class.__name__}")


    def _resolve_pk_field_by_node_name(self, node_name: str) -> str:
        """Resolve the primary key field name for a node by its registered name.

        @@ STEP: Use registry lookup to avoid isinstance/hasattr on type objects
        || S.S.: Provide detailed error information for debugging registry issues
        """
        node_cls = get_node_by_name(node_name)
        if node_cls:
            pk_fields = node_cls.get_primary_key_fields()
            if pk_fields:
                return pk_fields[0]
            else:
                raise ValueError(
                    f"Node class {node_cls.__name__} found in registry but has no primary key fields. "
                    f"Ensure at least one field has primary_key=True in the model definition."
                )
        else:
            registered_nodes = list(get_registered_nodes().keys())
            raise ValueError(
                f"Node '{node_name}' not found in registry. "
                f"Available registered nodes: {registered_nodes}. "
                f"Ensure the node class is decorated with @kuzu_node('{node_name}') and imported."
            )


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
            raise ValueError("Cannot update instance without primary key")

        # Build UPDATE query
        properties = instance.model_dump(exclude_unset=True)

        # Remove primary key from properties to update
        cached_pk_fields = self._get_cached_metadata(model_class, 'pk_fields')

        if cached_pk_fields is not None:
            pk_fields = cached_pk_fields
        elif hasattr(model_class, 'get_primary_key_fields'):
            get_pk_fields = model_class.get_primary_key_fields
            if not callable(get_pk_fields):
                raise ValueError(
                    f"Model {model_class.__name__}.get_primary_key_fields exists but is not callable"
                )
            pk_fields = cast(List[str], get_pk_fields())
            # Cache the result for future use
            self._set_cached_metadata(model_class, 'pk_fields', pk_fields)
        else:
            pk_fields = ['id']
            # Cache the default for future use
            self._set_cached_metadata(model_class, 'pk_fields', pk_fields)

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
            self._execute_with_connection_reuse(query, params)

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
            raise ValueError("Cannot delete instance without primary key")

        # Build DELETE query
        cached_pk_fields = self._get_cached_metadata(model_class, 'pk_fields')

        if cached_pk_fields is not None:
            pk_fields = cached_pk_fields
        elif hasattr(model_class, 'get_primary_key_fields'):
            get_pk_fields = model_class.get_primary_key_fields
            if not callable(get_pk_fields):
                raise ValueError(
                    f"Model {model_class.__name__}.get_primary_key_fields exists but is not callable"
                )
            pk_fields = cast(List[str], get_pk_fields())
            # Cache the result for future use
            self._set_cached_metadata(model_class, 'pk_fields', pk_fields)
        else:
            pk_fields = ['id']
            # Cache the default for future use
            self._set_cached_metadata(model_class, 'pk_fields', pk_fields)

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
        self._execute_with_connection_reuse(query, params)

        # Remove from identity map with optimized key generation
        identity_key = self._generate_identity_key(model_class, pk_value)
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
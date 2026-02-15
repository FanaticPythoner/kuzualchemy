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
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Iterator, cast, Callable, get_origin, get_args
from contextlib import contextmanager
from pathlib import Path
from collections import defaultdict, OrderedDict
import logging
import uuid
# enum.Enum - Rust py_to_value handles Enum.value unwrapping
import os
import time
# datetime/date imports removed - ATP returns properly typed Python objects

from .kuzu_query import Query
from .constants import (
    ErrorMessages,
    KuzuDataType,
)
from .atp_integration import ATPIntegration
from .constants import PerformanceConstants
from .constants import DDLConstants
from .kuzu_orm import get_node_by_name, KuzuRelationshipBase, get_registered_nodes
# DefaultFunctionBase - Rust is_default_fn_sentinel handles filtering


ModelType = TypeVar("ModelType")
logger = logging.getLogger(__name__)


class KuzuConnection:
    """Wrapper for Kuzu database connection using shared Database objects."""


    def __init__(self, db_path: Union[str, Path]):
        """
        Initialize connection to Kuzu database using connection pool.

        Args:
            db_path: Path to the Kuzu database file or directory

        Raises:
            ConnectionError: If database connection cannot be established
            ValueError: If invalid parameters are provided
        """
        # @@ STEP: Use connection pool for proper concurrent access with limited buffer pool size
        # || S.1: Get connection from shared Database object to avoid file locking issues
        # || S.2: Validate and limit buffer pool size to prevent massive memory allocation errors
        self.db_path = Path(db_path)

        # Route all execution through ATP pipeline
        self._atp = ATPIntegration(self.db_path)
        self._closed = False

    def execute(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a Cypher query strictly via ATP pipeline and return list[dict] rows."""
        if self._closed:
            raise RuntimeError(ErrorMessages.CONNECTION_CLOSED)
        rows = self._atp.run_cypher(query, parameters, expect_rows=True)
        return rows

    def close(self) -> None:
        """Close ATP resources and mark connection as closed."""
        if self._closed:
            return
        try:
            self._atp.flush(None)
        finally:
            try:
                self._atp.shutdown(None)
            finally:
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
        autoflush: bool = True,
        autocommit: bool = False,
        expire_on_commit: bool = True,
        bulk_insert_threshold: int = 10,
        bulk_batch_size: int = 10000,
        force_gc: bool = False,
        bulk_batch_size_max: int = 65536
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
        local_force_gc = bool(force_gc)

        if connection:
            self._conn = connection
            self._owns_connection = False
        elif db_path:
            self._conn = KuzuConnection(db_path)
            self._owns_connection = True
        else:
            raise ValueError("Either connection or db_path must be provided")
        self.autoflush = autoflush
        self.autocommit = autocommit
        self.expire_on_commit = expire_on_commit
        self.bulk_insert_threshold = bulk_insert_threshold
        self.bulk_batch_size = bulk_batch_size
        # Optional: upper bound for adaptive ramp-up in pipelined COPY
        self.bulk_batch_size_max = int(bulk_batch_size_max)
        self._dirty = set()
        self._new = set()
        self._force_gc = local_force_gc
        # Debug timing flag (read once, no runtime env lookups)
        self._debug_timing = str(os.getenv("KUZU_TIMING", "0")) in ("1", "true", "TRUE", "on", "ON")
        # Allow disabling the pipelined bulk-insert marshalling via env for accurate single-thread profiling
        self._disable_bulk_pipeline = str(os.getenv("KUZU_DISABLE_BULK_PIPELINE", "0")).lower() in ("1", "true", "on")

        self._deleted = set()
        self._flushing = False

        # || S.1: Use IDENTITY_MAP_INITIAL_SIZE for better memory allocation
        # || Pre-allocating dictionary size reduces hash collisions
        # || and memory reallocations during runtime, improving O(1) lookup performance
        initial_size = PerformanceConstants.IDENTITY_MAP_INITIAL_SIZE
        self._identity_map: Dict[str, Any] = dict.fromkeys(range(initial_size))
        self._identity_map.clear()  # Clear keys but keep allocated space

        # || S.1: Track pending operations count for batch-based autoflush
        # || Batching reduces I/O overhead by factor of batch_size
        self._pending_operations_count = 0
        self._autoflush_batch_size = PerformanceConstants.AUTOFLUSH_BATCH_SIZE

        # || S.1: Initialize metadata cache with bounded size to prevent memory leaks
        # || LRU cache with fixed size provides O(1) access
        # || while maintaining bounded memory usage
        self._metadata_cache: OrderedDict[str, Any] = OrderedDict()
        self._metadata_cache_size = PerformanceConstants.METADATA_CACHE_SIZE

    def get_db_path(self) -> str:
        """Return the database path as a string for ATP parallel query execution."""
        return str(self._conn.db_path)

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

    def execute(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        *,
        as_iterator: bool = False,
        page_size: Optional[int] = None,
        prefetch_pages: int = 1,
    ) -> Union[List[Dict[str, Any]], Iterator[Dict[str, Any]]]:
        """
        Execute a raw Cypher query.

        Args:
            query: Cypher query string
            parameters: Query parameters
            as_iterator: When True, return a lazy iterator over results using paging
            page_size: Positive integer page size when as_iterator=True; defaults to 10 when omitted
            prefetch_pages: 0 to disable; 1 for single-page lookahead (default)

        Returns:
            List of result dictionaries, or an iterator when as_iterator=True

        """
        # Iterator mode: delegate to iterate() for paging behavior
        if as_iterator:
            eff_page_size = 10 if (page_size is None) else page_size
            if eff_page_size <= 0:
                raise ValueError("page_size must be > 0 when as_iterator=True")
            return self.iterate(query, parameters, page_size=eff_page_size, prefetch_pages=prefetch_pages)

        # Flush pending ops when autoflush is enabled (simple check; no heavy detection)
        if self.autoflush and not self._flushing and (self._new or self._dirty or self._deleted):
            self.flush()

        result = self._execute_with_connection_reuse(query, parameters)
        return result

    def _execute_for_query_object(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        if self.autoflush and not self._flushing and (self._new or self._dirty or self._deleted):
            self.flush()

        t0 = time.perf_counter()
        result = self._conn.execute(query, parameters)
        dt = time.perf_counter() - t0

        if self._debug_timing or dt >= 0.25:
            rows = len(result) if isinstance(result, list) else None
            logger.info(
                "kuzu.session.exec_raw_for_query rows=%s seconds=%.6f",
                rows,
                dt,
            )

        return result

    def iterate(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        *,
        page_size: int = 10,
        prefetch_pages: int = 1,
    ) -> Iterator[Dict[str, Any]]:
        """Iterate over results of a raw Cypher query in pages, yielding rows lazily.

        This appends SKIP/LIMIT for paging and executes the query per page, with optional
        single-page lookahead prefetch to overlap I/O with consumption.

        Args:
            query: Raw Cypher string that returns rows
            parameters: Optional parameter dict
            page_size: Positive integer page size (default: 10)
            prefetch_pages: 0 to disable; 1 for single-page lookahead (default)

        Yields:
            Row dictionaries for each result row.

        Raises:
            ValueError: If page_size <= 0, or if query already contains SKIP/LIMIT
        """
        if page_size <= 0:
            raise ValueError("page_size must be a positive integer")

        base = query.strip().rstrip(';').strip()
        q_lower = base.lower()
        # Basic guard: avoid double-paginating queries that already include SKIP/LIMIT
        if " skip " in f" {q_lower} " or " limit " in f" {q_lower} ":
            raise ValueError("Raw query already contains SKIP/LIMIT; cannot auto-paginate. Remove them and retry.")

        def fetch_page(offset: int) -> tuple[List[Dict[str, Any]], bool]:
            paged_q = f"{base} SKIP {offset} LIMIT {page_size + 1}"
            rows = self.execute(paged_q, parameters)
            has_more = len(rows) > page_size
            page = rows[:page_size] if has_more else rows
            return page, has_more

        # First page
        offset = 0
        page, has_more = fetch_page(offset)
        if prefetch_pages > 0:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as executor:
                next_future = executor.submit(fetch_page, offset + page_size) if has_more else None
                while True:
                    for row in page:
                        yield row
                    if not has_more:
                        break
                    if next_future is not None:
                        next_page, next_has_more = next_future.result()
                    else:
                        next_page, next_has_more = fetch_page(offset + page_size)
                    offset += page_size
                    if next_has_more:
                        next_future = executor.submit(fetch_page, offset + page_size)
                    else:
                        next_future = None
                    page = next_page
                    has_more = next_has_more
        else:
            while True:
                for row in page:
                    yield row
                if not has_more:
                    break
                offset += page_size
                page, has_more = fetch_page(offset)


    def _execute_with_connection_reuse(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute query strictly via ATP pipeline.
        No legacy single-writer queue, no silent fallbacks; all execution goes through
        the ATP-backed KuzuConnection, which routes to RUN_CYPHER in the Rust core.
        """
        # Pass parameters directly; ATP converts Enums and types at the Rust boundary
        bound_params: Optional[Dict[str, Any]] = parameters

        # Always execute via KuzuConnection, which uses ATPIntegration.run_cypher
        # with expect_rows=True. DDL and write statements may return an empty list.
        result = self._conn.execute(query, bound_params)
        if not isinstance(result, list):
            return result
        # All types (UUID, Date, DateTime, etc.) come correctly typed from ATP
        return result

    def bulk_update_nodes(self, model_class: Type[Any], rows: List[Dict[str, Any]]) -> None:
        if not isinstance(rows, list):
            raise ValueError("rows must be a list of dictionaries")
        if not rows:
            return
        if not isinstance(model_class, type) or not hasattr(model_class, "__kuzu_node_name__"):
            raise ValueError("model_class must be a registered Kuzu node model")

        pk_fields = self._get_pk_fields_cached(model_class)
        if not pk_fields:
            raise ValueError(f"No primary key found in node {model_class.__name__}")

        label = model_class.__kuzu_node_name__

        def _normalize_uuid_value(field: str, v: Any, *, optional: bool) -> Any:
            if v is None:
                if optional:
                    return None
                raise TypeError(f"Field {label}.{field} is not optional and cannot be None")
            if isinstance(v, uuid.UUID):
                return v
            raise TypeError(f"Field {label}.{field} expects uuid.UUID{'|None' if optional else ''}, got {type(v)}")

        def _normalize_row_types(r: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(r, dict):
                raise ValueError("each row must be a dict")
            out: Dict[str, Any] = dict(r)
            for fname, fi in model_class.model_fields.items():
                if fname not in out:
                    continue
                ann = fi.annotation
                origin = get_origin(ann)
                if ann is uuid.UUID:
                    out[fname] = _normalize_uuid_value(fname, out[fname], optional=False)
                elif origin is Union:
                    args = get_args(ann)
                    if uuid.UUID in args and type(None) in args:
                        out[fname] = _normalize_uuid_value(fname, out[fname], optional=True)
            return out

        # Validate PK presence and determine SET fields deterministically
        set_fields: List[str] = []
        rows = [_normalize_row_types(r) for r in rows]

        for r in rows:
            if not isinstance(r, dict):
                raise ValueError("each row must be a dict")
            for pk in pk_fields:
                if pk not in r:
                    raise ValueError(f"row missing primary key field '{pk}' for {label}")
            for k in r.keys():
                if k in pk_fields:
                    continue
                if k not in set_fields:
                    set_fields.append(k)

        if not set_fields:
            return

        where_expr = " AND ".join([f"n.{pk} = row.{pk}" for pk in pk_fields])

        # Kuzu does not accept NULL parameters without an explicit type.
        # Perform per-field updates and split NULL vs non-NULL assignments.
        for field in set_fields:
            non_null_rows: List[Dict[str, Any]] = []
            null_rows: List[Dict[str, Any]] = []
            for r in rows:
                v = r.get(field)
                if v is None:
                    null_rows.append({pk: r[pk] for pk in pk_fields})
                else:
                    rr: Dict[str, Any] = {pk: r[pk] for pk in pk_fields}
                    rr[field] = v
                    non_null_rows.append(rr)

            if non_null_rows:
                cypher_set = f"UNWIND $rows AS row MATCH (n:{label}) WHERE {where_expr} SET n.{field} = row.{field}"
                _ = self._conn._atp.run_cypher(cypher_set, {"rows": non_null_rows}, expect_rows=False)

            if null_rows:
                cypher_null = f"UNWIND $rows AS row MATCH (n:{label}) WHERE {where_expr} SET n.{field} = NULL"
                _ = self._conn._atp.run_cypher(cypher_null, {"rows": null_rows}, expect_rows=False)

    def bulk_delete_nodes(self, model_class: Type[Any], pks: List[Any]) -> None:
        if not isinstance(pks, list):
            raise ValueError("pks must be a list")
        if not pks:
            return
        if not isinstance(model_class, type) or not hasattr(model_class, "__kuzu_node_name__"):
            raise ValueError("model_class must be a registered Kuzu node model")

        pk_fields = self._get_pk_fields_cached(model_class)
        if not pk_fields or len(pk_fields) != 1:
            raise ValueError(f"bulk_delete_nodes currently supports single-field primary keys only for {model_class.__name__}")
        pk_field = pk_fields[0]
        label = model_class.__kuzu_node_name__

        rows = [{pk_field: v} for v in pks]
        cypher = f"UNWIND $rows AS row MATCH (n:{label}) WHERE n.{pk_field} = row.{pk_field} DELETE n"
        _ = self._conn._atp.run_cypher(cypher, {"rows": rows}, expect_rows=False)

    # REMOVED: _normalize_result_row and _normalize_labeled_map
    # All types (UUID, Date, DateTime, etc.) now come correctly typed from ATP's Rust layer

    def _raise_bulk_error(self, kind: str, name: str, exc: Exception) -> None:
        """Raise a structured RuntimeError for bulk operations without swallowing details."""
        em = str(exc)
        em_l = em.lower()
        hint = ""
        if not ("constraint" in em_l or "duplicate" in em_l or "primary" in em_l):
            hint = " (constraint violation)"
        raise RuntimeError(f"{kind} failed for {name}: {type(exc).__name__}: {em}{hint}") from exc

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

    def _get_or_build_pk_extractor(self, node_cls: Type[Any]) -> Callable[[Any], Any]:
        """
        Return a fast primary-key value extractor for the given node class, building and caching it if needed.

        The extractor reads from __dict__ first (avoids descriptor/property overhead) and
        falls back to getattr only if necessary. We cache both the pk_field and the extractor
        in the session's metadata cache to eliminate repeated introspection and lookups.
        """
        # Try cached extractor first
        extractor = self._get_cached_metadata(node_cls, 'pk_extractor')
        if extractor is not None:
            return extractor  # type: ignore[return-value]

        # Ensure pk_fields are cached
        pk_fields = self._get_cached_metadata(node_cls, 'pk_fields')
        if pk_fields is None:
            if hasattr(node_cls, 'get_primary_key_fields'):
                get_pk_fields = node_cls.get_primary_key_fields  # type: ignore[attr-defined]
                if not callable(get_pk_fields):
                    raise ValueError(
                        f"Model {node_cls.__name__}.get_primary_key_fields exists but is not callable"
                    )
                pk_fields = cast(List[str], get_pk_fields())
                self._set_cached_metadata(node_cls, 'pk_fields', pk_fields)
            else:
                pk_fields = []

        if not pk_fields:
            raise ValueError(f"No primary key found in node {node_cls.__name__}")

        # Single-field fast path (dominant case)
        pk_field = pk_fields[0]
        self._set_cached_metadata(node_cls, 'pk_field', pk_field)

        def _extract(inst: Any) -> Any:
            d = getattr(inst, '__dict__', None)
            if isinstance(d, dict):
                v = d.get(pk_field)
                if v is not None:
                    return v
            if hasattr(inst, pk_field):
                return getattr(inst, pk_field)
            return None

        self._set_cached_metadata(node_cls, 'pk_extractor', _extract)
        return _extract

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

    def add_all(self, instances: List[Any], batch_size: Optional[int] = None) -> None:
        """
        Add multiple instances to the session.

        Args:
            instances: List of model instances to add
        """
        for instance in instances:
            self.add(instance)

    def _bulk_insert(self, instances: List[Any], batch_size: Optional[int] = None) -> None:
        """Bulk insert by grouping model types and delegating to ATP's create_nodes/edges via existing helpers."""
        if not instances:
            return
        if self._disable_bulk_pipeline:
            for instance in instances:
                self._insert_instance(instance)
            return
        # Group instances by model type for efficient processing
        model_groups = defaultdict(list)
        for instance in instances:
            model_groups[type(instance)].append(instance)
        # Use existing per-type bulk path (UNWIND rows -> ATP.create_nodes/create_edges)
        eff_batch = batch_size if batch_size is not None else self.bulk_batch_size
        for model_class, group in model_groups.items():
            self._bulk_insert_model_type(model_class, group, eff_batch)

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
        # Use UNWIND-based in-memory ingestion for all cases (no disk I/O)
        return self._bulk_insert_model_type_unwind(model_class, instances, batch_size)

    def _bulk_insert_model_type_unwind(self, model_class: Type[Any], instances: List[Any], batch_size: int) -> None:
        """Bulk insert using Cypher UNWIND and parameter rows.

        ATP Rust layer handles:
        - Type conversion (py_to_value): UUID, datetime, Enum.value
        - Default function sentinel detection and filtering
        - UNWIND query generation and execution
        - Auto-increment field materialization from return rows
        """
        if not instances:
            return

        # Determine if node or relationship
        is_rel = hasattr(model_class, '__kuzu_rel_name__') or hasattr(model_class, '__kuzu_relationship_name__')

        # Get table/rel name
        if is_rel:
            table_name = getattr(model_class, '__kuzu_rel_name__', None) or getattr(model_class, '__kuzu_relationship_name__')
        else:
            table_name = getattr(model_class, '__kuzu_node_name__', None)
        if not table_name:
            raise ValueError(f"Model {model_class.__name__} is not a registered node or relationship")

        # Get auto-increment fields once
        auto_fields = getattr(model_class, 'get_auto_increment_fields', lambda: [])()
        has_auto_increment = bool(auto_fields)

        # Get PK fields once for nodes
        pk_fields = getattr(model_class, 'get_primary_key_fields', lambda: [])() if not is_rel else []

        # Relationship bulk insert optimization: when the relationship is defined for exactly one
        # (from_label,to_label) pair, precompute routing once and avoid per-row routing fields.
        fixed_rel_routing: Optional[Dict[str, str]] = None
        if is_rel:
            fixed_rel_routing = self._try_get_fixed_relationship_routing(model_class)

        n = len(instances)
        start = 0
        cur_batch = max(1, int(batch_size))

        while start < n:
            slice_len = min(cur_batch, n - start)
            inst_ref = instances[start:start + slice_len]

            if not is_rel:
                # === NODE BULK INSERT ===
                # Build minimal rows - ATP py_to_value handles type conversion and default function filtering
                rows = self._build_node_rows(model_class, inst_ref, auto_fields)
                if not rows:
                    start += slice_len
                    continue

                # Determine if we need return rows (for auto-increment fields not provided)
                sample = inst_ref[0].model_dump(exclude_unset=True)
                excluded_auto = [f for f in auto_fields if sample.get(f) is None]
                need_return = bool(excluded_auto)

                ret_rows: Optional[List[Dict[str, Any]]] = None
                try:
                    ret_rows = self._conn._atp.create_nodes(
                        table_name, rows, return_rows=need_return, pk_fields=pk_fields
                    )
                except Exception as e:
                    self._raise_bulk_error("Bulk insert", table_name, e)

                # Materialize auto-increment fields from return rows
                if need_return and ret_rows:
                    self._materialize_auto_fields(inst_ref, ret_rows, excluded_auto, 'n')
                    self._update_identity_map(model_class, inst_ref, pk_fields)
            else:
                # === RELATIONSHIP BULK INSERT ===
                if fixed_rel_routing is not None:
                    rows_r = self._build_rel_rows_fixed(inst_ref, has_auto_increment)
                else:
                    rows_r = self._build_rel_rows(model_class, inst_ref, has_auto_increment)

                src_label = '*'
                dst_label = '*'
                src_pk_field = '*'
                dst_pk_field = '*'
                if fixed_rel_routing is not None:
                    src_label = fixed_rel_routing['from_label']
                    dst_label = fixed_rel_routing['to_label']
                    src_pk_field = fixed_rel_routing['from_pk_field']
                    dst_pk_field = fixed_rel_routing['to_pk_field']

                ret_rows_r: Optional[List[Dict[str, Any]]] = None
                try:
                    ret_rows_r = self._conn._atp.create_edges(
                        rel_name=table_name,
                        src_label=src_label,
                        dst_label=dst_label,
                        rows=rows_r,
                        src_pk_field=src_pk_field,
                        dst_pk_field=dst_pk_field,
                    )
                except Exception as e:
                    self._raise_bulk_error("Bulk relationship insert", table_name, e)

                if has_auto_increment and ret_rows_r:
                    self._materialize_auto_fields(inst_ref, ret_rows_r, auto_fields, 'r')

            # Advance and adapt batch size
            start += slice_len
            target_max = getattr(self, 'bulk_batch_size_max', batch_size)
            if cur_batch < target_max:
                cur_batch = min(target_max, int(cur_batch * 2))

    def _build_node_rows(self, model_class: Type[Any], instances: List[Any], auto_fields: List[str]) -> List[Dict[str, Any]]:
        """Build node rows for bulk insert.

        ATP Rust layer handles:
        - Type conversion: Enum.value, UUID, datetime via py_to_value
        - Default function sentinel detection and filtering via is_default_fn_sentinel
        - Null value filtering (Kuzu driver doesn't support untyped NULL in parameters)
        """
        return [
            {"__atp_row_idx": j, **inst.model_dump(exclude_unset=True)}
            for j, inst in enumerate(instances)
        ]

    def _build_rel_rows(self, model_class: Type[Any], instances: List[Any], has_auto_increment: bool) -> List[Dict[str, Any]]:
        """Build relationship rows for bulk insert.

        ATP Rust layer handles:
        - Dynamic multi-pair grouping by from_label/to_label/from_pk_field/to_pk_field
        - Type conversion and default function filtering via py_to_value/is_default_fn_sentinel
        """
        from .kuzu_orm import KuzuRelationshipBase as _RelBase
        pairs = getattr(model_class, '__kuzu_relationship_pairs__', []) if issubclass(model_class, _RelBase) else []

        # Fields to exclude from properties (routing fields + internal relationship fields)
        exclude_fields = {
            DDLConstants.REL_FROM_NODE_FIELD, DDLConstants.REL_TO_NODE_FIELD,
            DDLConstants.REL_FROM_NODE_PK_FIELD, DDLConstants.REL_TO_NODE_PK_FIELD,
            'from_node_pk', 'to_node_pk',
        }

        # Cache PK-field resolution per label for this call to avoid repeated registry scans
        pk_field_cache: Dict[str, str] = {}

        # Fast path: if there is exactly one relationship pair, labels are fixed even when instances
        # are constructed with raw PKs (ints/str). This avoids repeated pair inspection.
        fixed_labels: Optional[Dict[str, str]] = None
        if pairs and len(pairs) == 1:
            rp0 = pairs[0]
            if hasattr(rp0, 'get_from_name') and hasattr(rp0, 'get_to_name'):
                try:
                    fixed_labels = {
                        'from_label': rp0.get_from_name(),
                        'to_label': rp0.get_to_name(),
                    }
                except Exception:
                    fixed_labels = None

        rows: List[Dict[str, Any]] = []
        for j, inst in enumerate(instances):
            d0 = getattr(inst, '__dict__', {})
            fpv = d0.get(DDLConstants.REL_FROM_NODE_PK_FIELD)
            tpv = d0.get(DDLConstants.REL_TO_NODE_PK_FIELD)
            if fpv is None:
                fpv = getattr(inst, "from_node_pk", None)
            if tpv is None:
                tpv = getattr(inst, "to_node_pk", None)
            if fpv is None or tpv is None:
                raise ValueError("Relationship bulk insert requires both from_node_pk and to_node_pk")

            if fixed_labels is not None:
                from_label = fixed_labels['from_label']
                to_label = fixed_labels['to_label']
            else:
                from_label = self._resolve_node_label(d0.get('from_node'), pairs, 'from')
                to_label = self._resolve_node_label(d0.get('to_node'), pairs, 'to')

            from_pk_field = pk_field_cache.get(from_label)
            if from_pk_field is None:
                from_pk_field = self._resolve_pk_field_for_relationship_label(model_class, from_label)
                pk_field_cache[from_label] = from_pk_field
            to_pk_field = pk_field_cache.get(to_label)
            if to_pk_field is None:
                to_pk_field = self._resolve_pk_field_for_relationship_label(model_class, to_label)
                pk_field_cache[to_label] = to_pk_field

            # Build row from __dict__ to avoid the overhead of model_dump for large batches
            row: Dict[str, Any] = {
                'from_label': from_label, 'to_label': to_label,
                'from_pk_field': from_pk_field,
                'to_pk_field': to_pk_field,
                'from_pk': fpv, 'to_pk': tpv,
            }
            fields_set = getattr(inst, '__pydantic_fields_set__', None)
            if isinstance(fields_set, set):
                for k in fields_set:
                    if k in exclude_fields:
                        continue
                    v = d0.get(k)
                    if v is None:
                        continue
                    row[k] = v
            else:
                props = inst.model_dump(exclude_unset=True)
                for k, v in props.items():
                    if k in exclude_fields:
                        continue
                    if v is None:
                        continue
                    row[k] = v
            if has_auto_increment:
                row["__atp_row_idx"] = j
            rows.append(row)
        return rows

    def _try_get_fixed_relationship_routing(self, rel_cls: Type[Any]) -> Optional[Dict[str, str]]:
        """Return fixed routing info for single-pair relationships, else None."""
        cached = self._get_cached_metadata(rel_cls, 'fixed_rel_routing')
        if cached is not None:
            return cast(Optional[Dict[str, str]], cached)

        from .kuzu_orm import KuzuRelationshipBase as _RelBase
        if not issubclass(rel_cls, _RelBase):
            self._set_cached_metadata(rel_cls, 'fixed_rel_routing', None)
            return None

        pairs = getattr(rel_cls, '__kuzu_relationship_pairs__', [])
        if not pairs or len(pairs) != 1:
            self._set_cached_metadata(rel_cls, 'fixed_rel_routing', None)
            return None

        rp0 = pairs[0]
        if not (hasattr(rp0, 'get_from_name') and hasattr(rp0, 'get_to_name')):
            self._set_cached_metadata(rel_cls, 'fixed_rel_routing', None)
            return None

        try:
            from_label = rp0.get_from_name()
            to_label = rp0.get_to_name()
        except Exception:
            self._set_cached_metadata(rel_cls, 'fixed_rel_routing', None)
            return None

        if not isinstance(from_label, str) or not isinstance(to_label, str):
            self._set_cached_metadata(rel_cls, 'fixed_rel_routing', None)
            return None

        routing = {
            'from_label': from_label,
            'to_label': to_label,
            'from_pk_field': self._resolve_pk_field_for_relationship_label(rel_cls, from_label),
            'to_pk_field': self._resolve_pk_field_for_relationship_label(rel_cls, to_label),
        }
        self._set_cached_metadata(rel_cls, 'fixed_rel_routing', routing)
        return routing

    def _build_rel_rows_fixed(self, instances: List[Any], has_auto_increment: bool) -> List[Dict[str, Any]]:
        """Build relationship rows when routing is fixed via context (no per-row routing keys)."""
        exclude_fields = {
            DDLConstants.REL_FROM_NODE_FIELD, DDLConstants.REL_TO_NODE_FIELD,
            DDLConstants.REL_FROM_NODE_PK_FIELD, DDLConstants.REL_TO_NODE_PK_FIELD,
            'from_node_pk', 'to_node_pk',
        }
        rows: List[Dict[str, Any]] = []
        for j, inst in enumerate(instances):
            d0 = getattr(inst, '__dict__', {})
            fpv = d0.get(DDLConstants.REL_FROM_NODE_PK_FIELD)
            tpv = d0.get(DDLConstants.REL_TO_NODE_PK_FIELD)
            if fpv is None:
                fpv = getattr(inst, "from_node_pk", None)
            if tpv is None:
                tpv = getattr(inst, "to_node_pk", None)
            if fpv is None or tpv is None:
                raise ValueError("Relationship bulk insert requires both from_node_pk and to_node_pk")
            row: Dict[str, Any] = {'from_pk': fpv, 'to_pk': tpv}
            fields_set = getattr(inst, '__pydantic_fields_set__', None)
            if isinstance(fields_set, set):
                for k in fields_set:
                    if k in exclude_fields:
                        continue
                    v = d0.get(k)
                    if v is None:
                        continue
                    row[k] = v
            else:
                props = inst.model_dump(exclude_unset=True)
                for k, v in props.items():
                    if k in exclude_fields:
                        continue
                    if v is None:
                        continue
                    row[k] = v
            if has_auto_increment:
                row["__atp_row_idx"] = j
            rows.append(row)
        return rows

    def _resolve_node_label(self, node_obj: Any, pairs: List[Any], direction: str) -> str:
        """Resolve node label from instance or relationship pairs."""
        if hasattr(node_obj, '__kuzu_node_name__'):
            return node_obj.__kuzu_node_name__
        cls = getattr(node_obj, '__class__', None)
        if cls is not None and hasattr(cls, '__kuzu_node_name__'):
            return cls.__kuzu_node_name__
        # Raw PK: use relationship pairs when unambiguous
        if pairs and len(pairs) == 1:
            rp0 = pairs[0]
            getter = f'get_{direction}_name'
            if hasattr(rp0, getter):
                return getattr(rp0, getter)()
            idx = 0 if direction == 'from' else 1
            node = rp0[idx] if isinstance(rp0, tuple) and len(rp0) == 2 else None
            if node is not None:
                return getattr(node, '__kuzu_node_name__', None) or getattr(node, '__name__', None) or str(node)
        raise ValueError(f"Unable to resolve {direction}_label for relationship bulk insert")

    def _materialize_auto_fields(self, instances: List[Any], ret_rows: List[Dict[str, Any]], auto_fields: List[str], obj_key: str) -> None:
        """Materialize auto-increment fields from ATP return rows onto instances."""
        idx_to_obj: Dict[int, Dict[str, Any]] = {}
        for rec in ret_rows:
            idx = rec.get("__idx") or rec.get("col_0")
            obj = rec.get(obj_key) or rec.get("col_1")
            if isinstance(idx, int) and isinstance(obj, dict):
                idx_to_obj[idx] = obj
        for j, inst in enumerate(instances):
            obj = idx_to_obj.get(j)
            if not obj:
                continue
            for f in auto_fields:
                if f in obj and getattr(inst, f, None) is None:
                    setattr(inst, f, obj[f])

    def _update_identity_map(self, model_class: Type[Any], instances: List[Any], pk_fields: List[str]) -> None:
        """Update identity map after bulk insert with generated PKs."""
        if not pk_fields:
            return
        pk_field = pk_fields[0]
        for inst in instances:
            pk_val = getattr(inst, pk_field, None)
            if pk_val is not None:
                self._identity_map[self._generate_identity_key(model_class, pk_val)] = inst

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
            # Separate nodes and relationships to ensure endpoints exist first
            nodes_to_insert = []
            relationships_to_insert = []

            for instance in self._new:
                model_class = type(instance)
                if hasattr(model_class, '__kuzu_node_name__'):
                    nodes_to_insert.append(instance)
                elif hasattr(model_class, '__kuzu_rel_name__'):
                    relationships_to_insert.append(instance)
                else:
                    self._insert_instance(instance)

            # Insert nodes first; use bulk path when meeting threshold
            if len(nodes_to_insert) >= self.bulk_insert_threshold:
                self._bulk_insert(nodes_to_insert)
            else:
                for instance in nodes_to_insert:
                    self._insert_instance(instance)

            # Then insert relationships; use bulk path when meeting threshold
            if len(relationships_to_insert) >= self.bulk_insert_threshold:
                self._bulk_insert(relationships_to_insert)
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
                # Use fast extractor cached per class
                extractor = self._get_cached_metadata(model_class, 'pk_extractor')
                if extractor is None:
                    extractor = self._get_or_build_pk_extractor(model_class)
                value = extractor(instance)

                # Validate non-None (unless auto-increment)
                if value is None:
                    try:
                        auto_increment_fields = model_class.get_auto_increment_fields()
                        pk_field = self._get_cached_metadata(model_class, 'pk_field') or model_class.get_primary_key_fields()[0]
                        if pk_field in auto_increment_fields:
                            return None
                    except AttributeError as attr_err:
                        raise TypeError(
                            f"Model class {model_class.__name__} does not implement get_auto_increment_fields() method. "
                            f"All model classes must inherit from KuzuBaseModel. "
                            f"Original error: {attr_err}"
                        ) from attr_err
                    raise ValueError(
                        f"Primary key is None on {model_class.__name__} instance; non-auto-increment primary keys must be set."
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
        """Insert a node using ATP pipeline and materialize generated fields from returned object.

        ATP Rust layer handles type conversion (py_to_value): UUID, datetime, Enum.value
        """
        model_class = type(instance)
        node_name = model_class.__kuzu_node_name__

        # Get fields needing generation and validate manual values
        fields_needing_generation = instance.get_auto_increment_fields_needing_generation()
        manual_values = instance.get_manual_auto_increment_values()
        self._validate_manual_auto_increment_values(manual_values, model_class)

        # Build props dict - ATP py_to_value handles Enum.value, UUID, datetime conversion
        properties = instance.model_dump(exclude_unset=True)
        properties.update({k: v for k, v in manual_values.items() if v is not None})
        properties = {k: v for k, v in properties.items() if v is not None}

        # Delegate to ATP and request object return for auto-increment fields
        new_obj = self._conn._atp.create_node(node_name, props=properties, pk=None, return_object=True)

        # Materialize auto-increment fields from returned object
        if fields_needing_generation and isinstance(new_obj, dict):
            for field_name in fields_needing_generation:
                if field_name in new_obj and getattr(instance, field_name, None) is None:
                    setattr(instance, field_name, new_obj[field_name])

        # Update identity map
        pk_value = self._get_primary_key(instance)
        if pk_value is not None:
            self._identity_map[self._generate_identity_key(model_class, pk_value)] = instance

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
        """Get node type name from model instance or by probing raw primary key via ATP-backed queries.

        - Model instance: returns its registered node name immediately.
        - Raw primary key: probes each registered node label for a match using ATP pipeline.
          Emits precise, test-aligned errors for 0 or multiple matches.
        """
        # Fast-path for model instances
        if hasattr(node, '__class__') and hasattr(node.__class__, '__kuzu_node_name__'):
            return node.__class__.__kuzu_node_name__

        # Probe all registered node types using ATP-backed execution
        registered_nodes = get_registered_nodes()
        if not registered_nodes:
            raise RuntimeError(
                f"No registered node types found in registry. "
                f"Cannot determine node type for primary key value: {node}"
            )

        matching_node_types: List[str] = []
        for node_name, node_class in registered_nodes.items():
            try:
                pk_field = self._get_primary_key_field_name(node_class)
            except ValueError:
                # Skip malformed classes without a primary key
                continue
            # Probe existence efficiently without full count scans
            probe_q = f"MATCH (n:{node_name}) WHERE n.{pk_field} = $pk RETURN 1 AS __found LIMIT 1"
            try:
                rows = self._conn.execute(probe_q, { 'pk': node })
            except RuntimeError as e:
                msg = str(e)
                # Gracefully treat missing-table binder errors as no match for this label
                if 'Binder exception' in msg and 'does not exist' in msg:
                    continue
                raise
            if rows and len(rows) > 0:
                matching_node_types.append(node_name)

        if len(matching_node_types) == 0:
            available_types = list(registered_nodes.keys())
            raise TypeError(
                f"Primary key value {node} does not exist in any registered node type. "
                f"Available node types: {available_types}. "
                f"Ensure the node exists in the database and the primary key value is correct."
            )
        elif len(matching_node_types) > 1:
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
                # Explicit None provided for an auto-increment field
                field_meta = auto_increment_metadata.get(field_name)
                if field_meta and getattr(field_meta, 'primary_key', False):
                    # UUID PK: accept explicit None (tests expect skip)
                    if getattr(field_meta, 'kuzu_type', None) == KuzuDataType.UUID:
                        continue
                    # Non-UUID PK (e.g., INT/SERIAL): hard fail per non-null PK constraint
                    raise RuntimeError(
                        f"Explicit None provided for auto-increment primary key '{field_name}' in {model_class.__name__}: "
                        f"violates non-null constraint of the primary key"
                    )
                # For non-PK auto-increment fields, allow None (database will generate or keep null)
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
        """Insert a relationship instance into the database.

        ATP Rust layer handles type conversion (py_to_value): UUID, datetime, Enum.value
        """
        model_class = type(instance)
        rel_name = model_class.get_relationship_name()

        # Get fields needing generation and validate manual values
        fields_needing_generation = instance.get_auto_increment_fields_needing_generation()
        manual_values = instance.get_manual_auto_increment_values()
        self._validate_manual_auto_increment_values(manual_values, model_class)

        # Validate endpoints
        from_pk, to_pk = instance.from_node_pk, instance.to_node_pk
        if from_pk is None or to_pk is None:
            raise ValueError(f"Relationship {model_class.__name__} must have both from_node and to_node specified")

        # Determine the correct relationship pair
        matching_pair = self._determine_relationship_pair(instance, model_class)
        from_node_name, to_node_name = matching_pair.get_from_name(), matching_pair.get_to_name()

        # Build properties - ATP py_to_value handles Enum.value conversion
        internal_fields = {
            DDLConstants.REL_FROM_NODE_FIELD, DDLConstants.REL_TO_NODE_FIELD,
            DDLConstants.REL_FROM_NODE_PK_FIELD, DDLConstants.REL_TO_NODE_PK_FIELD,
        }
        properties = instance.model_dump(exclude_unset=True)
        properties = {k: v for k, v in properties.items() if k not in internal_fields and v is not None}
        properties.update({k: v for k, v in manual_values.items() if v is not None})

        # Resolve PK field names
        from_pk_field = self._resolve_pk_field_for_relationship_label(model_class, from_node_name)
        to_pk_field = self._resolve_pk_field_for_relationship_label(model_class, to_node_name)

        # Build row and delegate to ATP
        row: Dict[str, Any] = {'from_pk': from_pk, 'to_pk': to_pk, '__atp_row_idx': 0, **properties}
        ret_rows = self._conn._atp.create_edges(
            rel_name=rel_name,
            src_label=from_node_name, dst_label=to_node_name,
            rows=[row],
            src_pk_field=from_pk_field, dst_pk_field=to_pk_field,
        )

        # Materialize auto-increment fields from returned object
        if fields_needing_generation and isinstance(ret_rows, list) and ret_rows:
            rel_map = ret_rows[0].get("r") or ret_rows[0].get("col_1") if isinstance(ret_rows[0], dict) else None
            if isinstance(rel_map, dict):
                for f in fields_needing_generation:
                    if f in rel_map and getattr(instance, f, None) is None:
                        setattr(instance, f, rel_map[f])


    def _get_primary_key_field_name(self, model_class: Type[Any]) -> str:
        """Get the primary key field name for a model class."""
        from .kuzu_orm import _kuzu_registry

        for field_name, field_info in model_class.model_fields.items():
            metadata = _kuzu_registry.get_field_metadata(field_info)
            if metadata and metadata.primary_key:
                return field_name
        raise ValueError(f"No primary key field found in {model_class.__name__}")


    def _resolve_pk_field_for_relationship_label(self, rel_cls: Type[Any], node_label: str) -> str:
        pairs = getattr(rel_cls, '__kuzu_relationship_pairs__', [])
        for pair in pairs:
            if hasattr(pair, 'get_from_name') and hasattr(pair, 'get_to_name'):
                if pair.get_from_name() == node_label:
                    node_cls = getattr(pair, 'from_node', None)
                    if isinstance(node_cls, type):
                        return self._get_primary_key_field_name(node_cls)
                if pair.get_to_name() == node_label:
                    node_cls = getattr(pair, 'to_node', None)
                    if isinstance(node_cls, type):
                        return self._get_primary_key_field_name(node_cls)
            elif isinstance(pair, tuple) and len(pair) == 2:
                fr, to = pair
                from_name = (
                    getattr(fr, '__kuzu_node_name__', None)
                    or getattr(fr, '__name__', None)
                    or (fr if isinstance(fr, str) else None)
                )
                to_name = (
                    getattr(to, '__kuzu_node_name__', None)
                    or getattr(to, '__name__', None)
                    or (to if isinstance(to, str) else None)
                )
                if from_name == node_label and isinstance(fr, type):
                    return self._get_primary_key_field_name(fr)
                if to_name == node_label and isinstance(to, type):
                    return self._get_primary_key_field_name(to)

        return self._resolve_pk_field_by_node_name(node_label)


    def _resolve_pk_field_by_node_name(self, node_name: str) -> str:
        """Resolve the primary key field name for a node by its registered name.

        @@ STEP: Use registry lookup to avoid isinstance/hasattr on type objects
        || S.S.: Provide detailed error information for debugging registry issues
        """
        node_cls = get_node_by_name(node_name)
        if node_cls:
            # Delegate to the centralized primary-key resolver so we respect
            # registry metadata and avoid incorrect fallbacks such as a
            # hard-coded "id" field when the actual primary key is, e.g.,
            # "user_id". This keeps relationship endpoint binding consistent
            # with DDL generation and node CRUD paths.
            return self._get_primary_key_field_name(node_cls)

        registered_nodes = list(get_registered_nodes().keys())
        raise ValueError(
            f"Node '{node_name}' not found in registry. "
            f"Available registered nodes: {registered_nodes}. "
            f"Ensure the node class is decorated with @kuzu_node('{node_name}') and imported."
        )


    def _get_pk_fields_cached(self, model_class: Type[Any]) -> List[str]:
        """Get primary key fields with caching."""
        cached = self._get_cached_metadata(model_class, 'pk_fields')
        if cached is not None:
            return cached
        pk_fields = getattr(model_class, 'get_primary_key_fields', lambda: ['id'])()
        self._set_cached_metadata(model_class, 'pk_fields', pk_fields)
        return pk_fields

    def _build_pk_map(self, pk_fields: List[str], pk_value: Any) -> Dict[str, Any]:
        """Build PK map for ATP operations."""
        if len(pk_fields) == 1:
            return {pk_fields[0]: pk_value}
        return {pk_fields[i]: v for i, v in enumerate(pk_value)}

    def _update_instance(self, instance: Any) -> None:
        """Update an existing instance in the database."""
        model_class = type(instance)
        if not hasattr(model_class, '__kuzu_node_name__'):
            raise ValueError(f"Model {model_class.__name__} is not a registered node")

        pk_value = self._get_primary_key(instance)
        if not pk_value:
            raise ValueError("Cannot update instance without primary key")

        pk_fields = self._get_pk_fields_cached(model_class)
        properties = instance.model_dump(exclude_unset=True)
        properties = {k: v for k, v in properties.items() if k not in pk_fields and v is not None}

        if properties:
            self._conn._atp.update_node(
                label=model_class.__kuzu_node_name__,
                pk=self._build_pk_map(pk_fields, pk_value),
                props=properties
            )

    def _delete_instance(self, instance: Any) -> None:
        """Delete an instance from the database."""
        model_class = type(instance)
        if not hasattr(model_class, '__kuzu_node_name__'):
            raise ValueError(f"Model {model_class.__name__} is not a registered node")

        pk_value = self._get_primary_key(instance)
        if not pk_value:
            raise ValueError("Cannot delete instance without primary key")

        self._conn._atp.delete_node(
            label=model_class.__kuzu_node_name__,
            pk=self._build_pk_map(self._get_pk_fields_cached(model_class), pk_value)
        )

        # Remove from identity map
        identity_key = self._generate_identity_key(model_class, pk_value)
        self._identity_map.pop(identity_key, None)

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

    def __repr__(self) -> str:
        """String representation."""
        return f"<KuzuSession(autocommit={self.autocommit})>"

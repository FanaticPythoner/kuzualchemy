# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
Main Query class with method chaining for Kuzu ORM.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING, Any, Dict, List, Optional, Type, Union, TypeVar, Tuple, Iterator, Generic,
    get_origin, get_args,
)
import logging
import os
import threading
import uuid as uuid_module
from concurrent.futures import ThreadPoolExecutor
import time

from pydantic_core import PydanticUndefined


from .constants import ValidationMessageConstants, DDLConstants, QueryReturnAliasConstants
from .kuzu_query_expressions import (
    FilterExpression, AggregateFunction, OrderDirection, JoinType,
)
from .kuzu_query_builder import QueryState, JoinClause, CypherQueryBuilder
from .kuzu_query_fields import QueryField, ModelFieldAccessor
from .uuid_normalization import _NULL_UUID

logger = logging.getLogger(__name__)

_ENV_ATP_READONLY_POOL_MAX_SIZE = "ATP_READONLY_POOL_MAX_SIZE"


def _read_required_positive_int_env(var_name: str) -> int:
    """Read and validate a required positive integer environment variable."""
    raw = os.getenv(var_name)
    if raw is None:
        raise RuntimeError(
            f"Missing required environment variable '{var_name}'. "
            "Configure it before calling Query.iter()."
        )
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable '{var_name}' must be an integer, got: {raw!r}"
        ) from exc
    if value <= 0:
        raise ValueError(
            f"Environment variable '{var_name}' must be > 0, got: {value}"
        )
    return value

if TYPE_CHECKING:
    from .kuzu_session import KuzuSession

ModelType = TypeVar("ModelType")

# =============================================================================
# MODULE-LEVEL THREAD-SAFE CACHES FOR RELATIONSHIP MAPPING PERFORMANCE
# =============================================================================
# These caches persist across _map_results calls to avoid rebuilding lookups
# on every page/batch of relationship query results. Thread-safe via RLock.

# Cache: relationship_class -> (from_label_to_cls, to_label_to_cls)
_REL_LABEL_LOOKUP_CACHE: Dict[Type[Any], Tuple[Dict[str, Type[Any]], Dict[str, Type[Any]]]] = {}
_REL_LABEL_LOOKUP_LOCK = threading.RLock()

# Cache: node_class -> metadata dict with pk_fields, valid_fields, uuid_fields, etc.
_NODE_CLASS_META_CACHE: Dict[Type[Any], Dict[str, Any]] = {}
_NODE_CLASS_META_LOCK = threading.RLock()


def _get_node_class_meta(node_cls: Type[Any]) -> Dict[str, Any]:
    """Get or compute node class metadata (pk_fields, uuid_fields, etc.). Thread-safe."""
    # Fast path: check without lock first
    cached = _NODE_CLASS_META_CACHE.get(node_cls)
    if cached is not None:
        return cached

    # Slow path: acquire lock and compute
    with _NODE_CLASS_META_LOCK:
        # Double-check after acquiring lock
        cached = _NODE_CLASS_META_CACHE.get(node_cls)
        if cached is not None:
            return cached

        # Compute once per class
        pk_getter = getattr(node_cls, "get_primary_key_fields", None)
        if pk_getter is None:
            raise ValueError(f"Node class {node_cls.__name__} missing get_primary_key_fields")
        pk_fields = pk_getter()
        if type(pk_fields) is not list or len(pk_fields) == 0:
            raise ValueError(f"Node class {node_cls.__name__} returned invalid primary key fields")

        valid_fields = frozenset(node_cls.model_fields.keys())
        uuid_fields: set[str] = set()
        uuid_list_fields: set[str] = set()
        optional_uuid_fields: set[str] = set()
        fields_with_defaults: set[str] = set()

        for fname, fi in node_cls.model_fields.items():
            ann = fi.annotation
            origin = get_origin(ann)

            # Check scalar UUID
            if ann is uuid_module.UUID:
                uuid_fields.add(fname)

            # Check List[UUID]
            elif origin is list:
                args = get_args(ann)
                if len(args) == 1 and args[0] is uuid_module.UUID:
                    uuid_list_fields.add(fname)

            # Check Optional[UUID]
            elif origin is Union:
                args = get_args(ann)
                if (uuid_module.UUID in args) and (type(None) in args):
                    optional_uuid_fields.add(fname)
            # Check for defaults
            if (fi.default is not PydanticUndefined) or (fi.default_factory is not None):
                fields_with_defaults.add(fname)

        meta = {
            'pk_fields': pk_fields,
            'valid_fields': valid_fields,
            'uuid_fields': frozenset(uuid_fields),
            'uuid_list_fields': frozenset(uuid_list_fields),
            'optional_uuid_fields': frozenset(optional_uuid_fields),
            'fields_with_defaults': frozenset(fields_with_defaults),
        }
        _NODE_CLASS_META_CACHE[node_cls] = meta
        return meta


def _get_rel_label_lookups(
    rel_cls: Type[Any],
    from_candidates: List[Type[Any]],
    to_candidates: List[Type[Any]],
) -> Tuple[Dict[str, Type[Any]], Dict[str, Type[Any]]]:
    """Get or compute label->class lookup dicts for a relationship class. Thread-safe."""
    # Fast path: check without lock first
    cached = _REL_LABEL_LOOKUP_CACHE.get(rel_cls)
    if cached is not None:
        return cached

    # Slow path: acquire lock and compute
    with _REL_LABEL_LOOKUP_LOCK:
        # Double-check after acquiring lock
        cached = _REL_LABEL_LOOKUP_CACHE.get(rel_cls)
        if cached is not None:
            return cached

        # Build label -> class lookup dicts
        from_label_to_cls: Dict[str, Type[Any]] = {}
        for cls in from_candidates:
            d = cls.__dict__
            lbl = d['__kuzu_node_name__'] if '__kuzu_node_name__' in d else cls.__name__
            from_label_to_cls[lbl] = cls

        to_label_to_cls: Dict[str, Type[Any]] = {}
        for cls in to_candidates:
            d = cls.__dict__
            lbl = d['__kuzu_node_name__'] if '__kuzu_node_name__' in d else cls.__name__
            to_label_to_cls[lbl] = cls

        # Pre-warm node class metadata for all candidates
        for cls in from_candidates:
            _get_node_class_meta(cls)
        for cls in to_candidates:
            _get_node_class_meta(cls)

        _REL_LABEL_LOOKUP_CACHE[rel_cls] = (from_label_to_cls, to_label_to_cls)
        return from_label_to_cls, to_label_to_cls


class Query(Generic[ModelType]):
    """
    SQLAlchemy-like query builder for Kuzu ORM.
    Supports method chaining, filters, joins, aggregations, and more.
    """

    def __init__(
        self,
        model_class: Type[ModelType],
        session: Optional["KuzuSession"] = None,
        alias: str = "n"  # Default alias for node queries
    ):
        """Initialize query for a model class."""
        self._state = QueryState(model_class=model_class, alias=alias)
        self._session = session
        self._fields = ModelFieldAccessor(model_class)

    @property
    def fields(self) -> ModelFieldAccessor:
        """Access to model fields for query building."""
        return self._fields

    def _copy_with_state(self, **kwargs) -> Query:
        """Create a new Query with updated state."""
        new_query = Query.__new__(Query)
        new_query._state = self._state.copy(**kwargs)
        new_query._session = self._session
        new_query._fields = self._fields
        return new_query

    def filter(self, *expressions: FilterExpression) -> Query:
        """Add filter expressions to the query."""
        new_filters = list(self._state.filters)
        new_filters.extend(expressions)
        return self._copy_with_state(filters=new_filters)

    def where(self, expression: FilterExpression) -> Query:
        """Alias for filter()."""
        return self.filter(expression)

    def filter_by(self, **kwargs) -> Query:
        """Filter by field equality conditions."""
        expressions = []
        # @@ STEP: Use the correct model class and alias for field resolution
        # || S.S: After traversal, use return_model_class and return_alias for subsequent filters
        target_model = self._state.return_model_class or self._state.model_class

        for field_name, value in kwargs.items():
            # Create field using model-based resolution; the builder will map model -> alias
            # NOTE: Do not pre-qualify with alias here; doing so would duplicate alias in Cypher
            field = QueryField(field_name, target_model)
            expressions.append(field == value)
        return self.filter(*expressions)

    def order_by(self, *fields: Union[str, Tuple[str, OrderDirection], QueryField]) -> Query:
        """Add ordering to the query."""
        new_order = list(self._state.order_by)

        for field in fields:
            if isinstance(field, str):
                new_order.append((field, OrderDirection.ASC))
            elif isinstance(field, tuple):
                new_order.append(field)
            elif isinstance(field, QueryField):
                new_order.append((field.field_name, OrderDirection.ASC))
            else:
                raise ValueError(ValidationMessageConstants.INVALID_ORDER_BY_ARGUMENT.format(field))

        return self._copy_with_state(order_by=new_order)

    def limit(self, count: int) -> Query:
        """Limit the number of results."""
        return self._copy_with_state(limit_value=count)

    def offset(self, count: int) -> Query:
        """Offset the results."""
        return self._copy_with_state(offset_value=count)

    def distinct(self) -> Query:
        """Return only distinct results."""
        return self._copy_with_state(distinct=True)

    def select(self, *fields: Union[str, QueryField]) -> Query:
        """Select specific fields to return."""
        field_names = []
        for field in fields:
            if isinstance(field, str):
                field_names.append(field)
            elif isinstance(field, QueryField):
                field_names.append(field.field_name)
            else:
                raise ValueError(ValidationMessageConstants.INVALID_SELECT_FIELD.format(field))

        return self._copy_with_state(select_fields=field_names)

    def pairs_subset(self, indices: List[int]) -> Query:
        """Restrict relationship queries to a subset of pair indices for memory safety."""
        if indices is None:
            return self
        # Strict validation: list of non-negative ints
        if not isinstance(indices, list) or any(not isinstance(i, int) for i in indices):
            raise ValueError("pairs_subset expects a list of integers")
        if any(i < 0 for i in indices):
            raise ValueError("pairs_subset indices must be non-negative")
        return self._copy_with_state(pairs_subset=list(indices))

    def return_raw(self) -> Query:
        """Return all bound aliases and columns as raw dictionaries (RETURN *).

        This is useful for advanced patterns where multiple aliases (e.g., node pairs)
        are needed from a single query. Mapping to model instances can then be done
        by downstream code if desired.
        """
        return self._copy_with_state(return_raw=True)

    def join(
        self,
        target_model_or_rel: Type[Any],
        condition_or_model: Optional[Any] = None,
        join_type: Union[JoinType, str] = JoinType.INNER,
        target_alias: Optional[str] = None,
        rel_alias: Optional[str] = None,
        conditions: Optional[List[FilterExpression]] = None,
        direction: Optional[Any] = None,
        pattern: Optional[str] = None,
        properties: Dict[str, Any] = {},
        min_hops: int = 1,
        max_hops: int = 1
    ) -> Query:
        """Join with another node through a relationship.

        Supports two calling patterns:
        1. join(TargetModel, relationship_class, ...)
        2. join(RelationshipClass, condition, ...)
        """
        # @@ STEP: Handle different join calling patterns with strict typing
        # || S.S.1: Determine if first arg is a relationship class (no hasattr)
        is_rel_class_first = (
            isinstance(target_model_or_rel, type) and
            ('__kuzu_rel_name__' in target_model_or_rel.__dict__)
        )
        if is_rel_class_first:
            # Pattern 2: join(RelationshipClass, condition, ...)
            relationship_class = target_model_or_rel
            target_model = None
            # Decide whether the second argument is a model or a condition
            is_model_second = (
                isinstance(condition_or_model, type) and
                (condition_or_model is not None) and
                ('__kuzu_node_name__' in condition_or_model.__dict__)
            )
            if (condition_or_model is not None) and (not is_model_second):
                # It's a condition expression
                conditions = [condition_or_model]
            else:
                # It's a model class or None
                target_model = condition_or_model
                conditions = conditions or []
            # Do not assign a relationship alias by default; only use if provided explicitly
            # Provide deterministic target alias when user didn't specify any
            if target_alias is None:
                # If target model known, base alias on it; otherwise derive from relationship
                target_alias = (
                    f"{target_model.__name__.lower()}_joined" if target_model is not None
                    else f"{relationship_class.__name__.lower()}_to"
                )
        else:
            # Pattern 1: join(TargetModel, relationship_class, ...)
            if not isinstance(target_model_or_rel, type):
                raise TypeError("join() first argument must be a model class or relationship class")
            target_model = target_model_or_rel
            relationship_class = (
                condition_or_model if (
                    isinstance(condition_or_model, type) and
                    (condition_or_model is not None) and
                    ('__kuzu_rel_name__' in condition_or_model.__dict__)
                ) else None
            )
            conditions = conditions or []

        # || S.S.2: Convert string join_type to enum
        if isinstance(join_type, str):
            join_type = (
                JoinType[join_type.upper()]
                if join_type.upper() in JoinType.__members__
                else JoinType.INNER
            )

        if target_model and not target_alias:
            target_alias = f"{target_model.__name__.lower()}_joined"
        # Do not assign a relationship alias by default; keep None unless provided explicitly

        join_clause = JoinClause(
            relationship_class=relationship_class,
            target_model=target_model,
            join_type=join_type,
            source_alias=self._state.alias,
            target_alias=target_alias,
            rel_alias=rel_alias,
            conditions=conditions,
            direction=direction,
            pattern=pattern,
            properties=properties,
            min_hops=min_hops,
            max_hops=max_hops
        )

        new_joins = list(self._state.joins)
        new_joins.append(join_clause)

        return self._copy_with_state(joins=new_joins)



    def outerjoin(
        self,
        target_model: Type[Any],
        relationship_class: Optional[Type[Any]] = None,
        **kwargs
    ) -> Query:
        """Left outer join (OPTIONAL MATCH in Cypher)."""
        return self.join(
            target_model,
            relationship_class,
            join_type=JoinType.OPTIONAL,
            **kwargs
        )

    def traverse(
        self,
        relationship_class: Type[Any],
        target_model: Type[Any],
        direction: str = "outgoing",
        alias: Optional[str] = None,
        rel_alias: Optional[str] = None,
        conditions: Optional[List[FilterExpression]] = None
    ) -> Query:
        """
        Traverse from current nodes through a relationship to target nodes.

        This creates a join that focuses the query on the target nodes.

        Args:
            relationship_class: The relationship class to traverse
            target_model: The target node model class
            direction: "outgoing", "incoming", or "both"
            alias: Alias for the target nodes
            rel_alias: Alias for the relationship
            conditions: Additional filter conditions

        Returns:
            New Query instance focused on the target nodes
        """
        # @@ STEP: Create join and then change focus to target model
        if not alias:
            alias = f"{target_model.__name__.lower()}_joined"

        # First, add the join
        joined_query = self.join(
            target_model,
            relationship_class,
            target_alias=alias,
            rel_alias=rel_alias,
            conditions=conditions,
            direction=direction
        )

        # @@ STEP: Change the query focus to the target model
        # IMPORTANT: Don't change the main alias as it affects the initial MATCH
        # Only set return_alias and return_model_class for proper result handling
        return joined_query._copy_with_state(
            return_alias=alias,  # Set return alias for result mapping
            return_model_class=target_model  # Set return model for result mapping
        )

    def outgoing(
        self,
        relationship_class: Type[Any],
        target_model: Type[Any],
        alias: Optional[str] = None,
        rel_alias: Optional[str] = None,
        conditions: Optional[List[FilterExpression]] = None
    ) -> Query:
        """Traverse outgoing relationships to target nodes."""
        return self.traverse(
            relationship_class, target_model, "outgoing", alias, rel_alias, conditions
        )

    def incoming(
        self,
        relationship_class: Type[Any],
        target_model: Type[Any],
        alias: Optional[str] = None,
        rel_alias: Optional[str] = None,
        conditions: Optional[List[FilterExpression]] = None
    ) -> Query:
        """Traverse incoming relationships to target nodes."""
        return self.traverse(
            relationship_class, target_model, "incoming", alias, rel_alias, conditions
        )

    def related(
        self,
        relationship_class: Type[Any],
        target_model: Type[Any],
        alias: Optional[str] = None,
        rel_alias: Optional[str] = None,
        conditions: Optional[List[FilterExpression]] = None
    ) -> Query:
        """Traverse relationships in both directions to target nodes."""
        return self.traverse(
            relationship_class, target_model, "both", alias, rel_alias, conditions
        )

    def group_by(self, *fields: Union[str, QueryField]) -> Query:
        """Group results by fields."""
        field_names = []
        for field in fields:
            if isinstance(field, str):
                field_names.append(field)
            elif isinstance(field, QueryField):
                field_names.append(field.field_name)
            else:
                raise ValueError(f"Invalid group_by field: {field}")

        return self._copy_with_state(group_by=field_names)

    def having(self, expression: FilterExpression) -> Query:
        """Add HAVING clause for aggregations."""
        return self._copy_with_state(having=expression)

    def aggregate(
        self,
        alias: str,
        func: AggregateFunction,
        field: Union[str, QueryField]
    ) -> Query:
        """Add an aggregation to the query."""
        field_name = field if isinstance(field, str) else field.field_name
        new_aggregations = dict(self._state.aggregations)
        new_aggregations[alias] = (func, field_name)
        return self._copy_with_state(aggregations=new_aggregations)

    def count(self, field: Optional[Union[str, QueryField]] = None, alias: str = "count") -> Query:
        """Add COUNT aggregation."""
        field_name = "*" if field is None else (
            field if isinstance(field, str) else field.field_name
        )
        return self.aggregate(alias, AggregateFunction.COUNT, field_name)

    def sum(self, field: Union[str, QueryField], alias: str = "sum") -> Query:
        """Add SUM aggregation."""
        return self.aggregate(alias, AggregateFunction.SUM, field)

    def avg(self, field: Union[str, QueryField], alias: str = "avg") -> Query:
        """Add AVG aggregation."""
        return self.aggregate(alias, AggregateFunction.AVG, field)

    def min(self, field: Union[str, QueryField], alias: str = "min") -> Query:
        """Add MIN aggregation."""
        return self.aggregate(alias, AggregateFunction.MIN, field)

    def max(self, field: Union[str, QueryField], alias: str = "max") -> Query:
        """Add MAX aggregation."""
        return self.aggregate(alias, AggregateFunction.MAX, field)

    def union(self, other: Query, all: bool = False) -> Query:
        """Union with another query."""
        new_unions = list(self._state.union_queries)
        new_unions.append((other, all))
        return self._copy_with_state(union_queries=new_unions)

    def union_all(self, other: Query) -> Query:
        """Union all with another query."""
        return self.union(other, all=True)

    def with_raw(self, cypher: str) -> Query:
        """Add raw WITH clause."""
        new_with = list(self._state.with_clauses)
        new_with.append(cypher)
        return self._copy_with_state(with_clauses=new_with)

    def subquery(self, alias: str, query: Query) -> Query:
        """Add a subquery."""
        new_subqueries = dict(self._state.subqueries)
        new_subqueries[alias] = query
        return self._copy_with_state(subqueries=new_subqueries)

    def to_cypher(self) -> Tuple[str, Dict[str, Any]]:
        """Build the Cypher query and parameters."""
        builder = CypherQueryBuilder(self._state)
        return builder.build()

    def _execute(self) -> Any:
        """Execute the query and return results."""
        if not self._session:
            raise RuntimeError("No session attached to query")

        cypher, params = self.to_cypher()
        # Log the actual Cypher query for tracing
        model_name = getattr(self._state.model_class, "__name__", str(self._state.model_class))
        cypher_preview = cypher[:500] + "..." if len(cypher) > 500 else cypher
        logger.debug(
            "kuzu.query.execute model=%s cypher_len=%d params_count=%d cypher=%s",
            model_name,
            len(cypher),
            len(params),
            cypher_preview,
        )
        return self._session._execute_for_query_object(cypher, params)

    def iter(self, page_size: int = 10, prefetch_pages: int = 1) -> Iterator[Union[ModelType, Dict[str, Any]]]:
        """Return an iterator that yields items across pages lazily.

        Uses parallel page fetching via Rust rayon when ATP_READONLY_POOL_MAX_SIZE > 1
        and result set is large enough to benefit from parallelization.

        Args:
            page_size: Number of items to fetch per page (default: 10).
            prefetch_pages: Prefetch lookahead pages (0 disables; 1 enables single-page lookahead).

        Returns:
            An iterator that yields one item at a time across all pages.
        """
        if page_size <= 0:
            raise ValueError("page_size must be a positive integer")
        if not self._session:
            raise RuntimeError("No session attached to query")

        ps = int(page_size)
        pf = 1 if prefetch_pages and prefetch_pages > 0 else 0

        pairs_subset_meta = getattr(self._state, "pairs_subset", None)
        
        model_name = getattr(self._state.model_class, "__name__", str(self._state.model_class))
        logger.info(
            "kuzu.query.iter.open rel=%s page_size=%s prefetch_pages=%s pairs_subset=%s",
            model_name,
            int(page_size),
            int(prefetch_pages),
            pairs_subset_meta,
        )

        # Check if parallel execution is available and beneficial
        pool_size = _read_required_positive_int_env(_ENV_ATP_READONLY_POOL_MAX_SIZE)
        parallel_threshold = pool_size
        use_parallel = pool_size > 1 and parallel_threshold > 0

        def fetch_page(offset: int) -> List[Union[ModelType, Dict[str, Any]]]:
            q = self.offset(offset).limit(ps)
            t0 = time.perf_counter()
            raw = q._execute()
            t1 = time.perf_counter()
            mapped = q._map_results(raw)
            t2 = time.perf_counter()

            if getattr(self._session, "_debug_timing", False) or ((t2 - t0) >= 0.25):
                raw_rows = len(raw) if isinstance(raw, list) else None
                mapped_rows = len(mapped) if isinstance(mapped, list) else None
                logger.info(
                    "kuzu.query.page rel=%s offset=%d page_size=%d raw_rows=%s mapped_rows=%s exec_seconds=%.6f map_seconds=%.6f total_seconds=%.6f pairs_subset=%s",
                    model_name,
                    int(offset),
                    int(ps),
                    raw_rows,
                    mapped_rows,
                    (t1 - t0),
                    (t2 - t1),
                    (t2 - t0),
                    pairs_subset_meta,
                )

            return mapped

        def fetch_page_with_lookahead(offset: int) -> Tuple[List[Union[ModelType, Dict[str, Any]]], bool]:
            """Fetch a page with one-row lookahead to avoid terminal out-of-range SKIP queries."""
            q = self.offset(offset).limit(ps + 1)
            t0 = time.perf_counter()
            raw = q._execute()
            t1 = time.perf_counter()
            mapped = q._map_results(raw)
            t2 = time.perf_counter()

            has_more = len(mapped) > ps
            page_data = mapped[:ps] if has_more else mapped

            if getattr(self._session, "_debug_timing", False) or ((t2 - t0) >= 0.25):
                raw_rows = len(raw) if isinstance(raw, list) else None
                mapped_rows = len(page_data) if isinstance(page_data, list) else None
                logger.info(
                    "kuzu.query.page.lookahead rel=%s offset=%d page_size=%d raw_rows=%s mapped_rows=%s has_more=%s exec_seconds=%.6f map_seconds=%.6f total_seconds=%.6f pairs_subset=%s",
                    model_name,
                    int(offset),
                    int(ps),
                    raw_rows,
                    mapped_rows,
                    has_more,
                    (t1 - t0),
                    (t2 - t1),
                    (t2 - t0),
                    pairs_subset_meta,
                )

            return page_data, has_more

        def fetch_pages_parallel(offsets: List[int]) -> List[List[Union[ModelType, Dict[str, Any]]]]:
            """Fetch multiple pages in parallel using Rust rayon via ATP pipeline."""
            if not offsets:
                return []
            
            # Build queries for each offset
            queries: List[Tuple[str, Dict[str, Any]]] = []
            for off in offsets:
                q = self.offset(off).limit(ps)
                cypher, params = q.to_cypher()
                queries.append((cypher, params))
            
            # Execute in parallel via Rust - no fallback, fail loudly on errors
            from atp_pipeline import execute_parallel_queries
            db_path = self._session.get_db_path()
            t0 = time.perf_counter()
            raw_results = execute_parallel_queries(db_path, queries)
            t1 = time.perf_counter()
            
            # Map results for each page
            mapped_pages: List[List[Union[ModelType, Dict[str, Any]]]] = []
            for i, raw in enumerate(raw_results):
                q = self.offset(offsets[i]).limit(ps)
                m0 = time.perf_counter()
                mapped = q._map_results(raw)
                m1 = time.perf_counter()
                if getattr(self._session, "_debug_timing", False) or ((m1 - m0) >= 0.25):
                    raw_rows = len(raw) if isinstance(raw, list) else None
                    mapped_rows = len(mapped) if isinstance(mapped, list) else None
                    logger.info(
                        "kuzu.query.page.parallel rel=%s offset=%d page_size=%d raw_rows=%s mapped_rows=%s exec_seconds=%.6f map_seconds=%.6f pairs_subset=%s",
                        model_name,
                        int(offsets[i]),
                        int(ps),
                        raw_rows,
                        mapped_rows,
                        (t1 - t0),
                        (m1 - m0),
                        pairs_subset_meta,
                    )
                mapped_pages.append(mapped)
            return mapped_pages

        # If parallel execution is enabled, preserve existing count-bounded parallel strategy.
        if use_parallel:
            offset = 0
            page = fetch_page(offset)
            offset += ps

            # If first page is not full, result set fits in one page.
            if len(page) < ps:
                for item in page:
                    yield item
                return

            total_rows = self.count_results()
            remaining_rows = max(total_rows - ps, 0)

            # Yield first page items
            for item in page:
                yield item

            if remaining_rows == 0:
                return

            # Parallel batch fetching
            batch_size = min(pool_size, parallel_threshold)
            while remaining_rows > 0:
                pages_in_batch = min(batch_size, (remaining_rows + ps - 1) // ps)
                batch_offsets = [offset + i * ps for i in range(pages_in_batch)]

                # Fetch batch in parallel
                batch_pages = fetch_pages_parallel(batch_offsets)

                # Yield results in requested page order
                for page_data in batch_pages:
                    for item in page_data:
                        yield item

                advanced_rows = pages_in_batch * ps
                offset += advanced_rows
                remaining_rows = max(remaining_rows - advanced_rows, 0)

            return

        # Sequential modes: use +1 lookahead to avoid issuing a terminal out-of-range page.
        offset = 0
        page, has_more = fetch_page_with_lookahead(offset)
        offset += ps

        if pf > 0:
            # Sequential with prefetch (original behavior)
            with ThreadPoolExecutor(max_workers=1) as executor:
                next_future = executor.submit(fetch_page_with_lookahead, offset) if has_more else None
                while True:
                    for item in page:
                        yield item
                    if not has_more:
                        break
                    if next_future is not None:
                        next_page, next_has_more = next_future.result()
                    else:
                        next_page, next_has_more = fetch_page_with_lookahead(offset)
                    offset += ps
                    if next_has_more and pf > 0:
                        next_future = executor.submit(fetch_page_with_lookahead, offset)
                    else:
                        next_future = None
                    page = next_page
                    has_more = next_has_more
        else:
            # Pure sequential (no prefetch)
            while True:
                for item in page:
                    yield item
                if not has_more:
                    break
                page, has_more = fetch_page_with_lookahead(offset)
                offset += ps

    def all(self, as_iterator: bool = False, page_size: Optional[int] = None, prefetch_pages: int = 1) -> Union[List[ModelType], List[Dict[str, Any]], Iterator[Union[ModelType, Dict[str, Any]]]]:
        """Execute query and return all results or an iterator over results with paging.

        Args:
            as_iterator: When True, return an iterator that yields items one-by-one across pages.
            page_size: Number of rows per page to fetch when as_iterator=True (defaults to 10 if omitted).
            prefetch_pages: Number of pages to prefetch ahead (0 disables prefetch; values >1 are treated as 1).

        Returns:
            Either a list of results (default) or an iterator over results when as_iterator=True.
        """
        if as_iterator:
            eff_page_size = 10 if (page_size is None) else page_size
            if eff_page_size <= 0:
                raise ValueError("When as_iterator=True, page_size must be a positive integer")
            # Only single-page lookahead is currently supported.
            pf = 1 if prefetch_pages and prefetch_pages > 0 else 0
            return self.iter(page_size=eff_page_size, prefetch_pages=pf)

        # Default eager behavior
        results = self._execute()
        return self._map_results(results)

    def first(self) -> Union[ModelType, Dict[str, Any], None]:
        """Execute query and return first result."""
        limited = self.limit(1)
        results = limited.all()
        return results[0] if results else None

    def one(self) -> Union[ModelType, Dict[str, Any]]:
        """Execute query and return exactly one result."""
        results = self.all()
        if len(results) == 0:
            raise ValueError("Query returned no results")
        if len(results) > 1:
            raise ValueError(f"Query returned {len(results)} results, expected 1")
        return results[0]

    def one_or_none(self) -> Union[ModelType, Dict[str, Any], None]:
        """Execute query and return one result or None."""
        results = self.all()
        if len(results) > 1:
            raise ValueError(f"Query returned {len(results)} results, expected 0 or 1")
        return results[0] if results else None

    def exists(self) -> bool:
        """Check if any results exist."""
        limited = self.limit(1)
        results = limited._execute()
        return len(results) > 0

    def count_results(self) -> int:
        """Count the number of results."""
        # ORDER BY columns are not valid after scalar COUNT aggregation in Kuzu.
        # Keep all filters/joins while stripping ORDER BY for the COUNT query only.
        count_query = self._copy_with_state(order_by=[]).count()
        result = count_query._execute()
        if type(result) is not list:
            logger.error("Count query returned non-list result type: %r", type(result))
            raise TypeError("Count query did not return a list of rows")
        if len(result) == 0:
            # A COUNT over an empty set is 0 by definition
            return 0
        first = result[0]
        if type(first) is not dict:
            logger.error("Count query first row is not a dict: %r", type(first))
            raise TypeError("Count query row is not a dict")
        if "count" not in first:
            logger.error("Count query row missing 'count' key: %r", first)
            raise KeyError("Count query result missing 'count' field")
        return first["count"]

    def _map_results(
        self, raw_results: List[Dict[str, Any]]
    ) -> Union[List[ModelType], List[Dict[str, Any]]]:
        """Map raw results to model instances or return raw dictionaries for special cases."""
        if self._state.return_raw:
            return raw_results

        # Strict input validation
        if type(raw_results) is not list:
            logger.error("map_results expects list[dict], got: %r", type(raw_results))
            raise TypeError("Query mapping expects a list of dictionaries")
        for i, row in enumerate(raw_results):
            if type(row) is not dict:
                logger.error("Row %d is not a dict: %r", i, type(row))
                raise TypeError("Each result row must be a dictionary")

        # @@ STEP: Determine which alias and model class to use for result mapping (strict)
        result_alias: str | None = self._state.return_alias or self._state.alias
        result_model_class: type[Any] = self._state.return_model_class or self._state.model_class
        if result_alias is None:
            logger.error("Result alias is None; invalid query state")
            raise ValueError("Result alias must be defined for result mapping")
        if not isinstance(result_model_class, type):
            logger.error("Result model class is not a type: %r", type(result_model_class))
            raise TypeError("Result model class must be a class type")

        # @@ STEP: For GROUP BY queries with aggregations, return raw dictionaries
        # || S.1: GROUP BY queries return grouped fields + aggregated values, not full instances
        if self._state.aggregations:
            return raw_results

        # @@ STEP: Determine which alias and model class to use for result mapping
        result_alias = self._state.return_alias or self._state.alias
        result_model_class = self._state.return_model_class or self._state.model_class

        # Detect relationship result class without using forbidden reflection helpers
        d_model = result_model_class.__dict__
        is_relationship = (
            '__is_kuzu_relationship__' in d_model and
            d_model['__is_kuzu_relationship__'] is True
        )

        if is_relationship:
            from .kuzu_orm import KuzuRelationshipBase
            # Build candidate classes and prepare mapping
            mapped: List[Any] = []

            t_map_start = time.perf_counter()
            endpoint_cache: Dict[Tuple[str, Tuple[Any, ...]], Any] = {}
            cache_hits = 0
            cache_misses = 0

            # Build candidate classes from relationship pairs (handles multi-pair and legacy)
            has_pairs_attr = '__kuzu_relationship_pairs__' in d_model
            rel_pairs = d_model['__kuzu_relationship_pairs__'] if has_pairs_attr else []
            legacy_from = d_model['__kuzu_from_node__'] if '__kuzu_from_node__' in d_model else None
            legacy_to = d_model['__kuzu_to_node__'] if '__kuzu_to_node__' in d_model else None
            logger.info(
                "RelClass=%s has_pairs_attr=%s pairs_len=%s legacy_from=%s legacy_to=%s",
                result_model_class.__name__, has_pairs_attr, len(rel_pairs),
                type(legacy_from).__name__ if legacy_from is not None else None,
                type(legacy_to).__name__ if legacy_to is not None else None,
            ) # TODO: REVERT TO debug()
            if (not rel_pairs) and (legacy_from is None or legacy_to is None):
                logger.error(
                    "Relationship class %s lacks pairs and legacy endpoints",
                    result_model_class.__name__
                )
                raise ValueError(
                    "Relationship class missing pair definitions: "
                    "either pairs or legacy endpoints must be defined"
                )

            def _node_class_or_none(x: Any) -> Optional[Type[Any]]:
                # Check if x is a class/type first (basic type safety)
                if not isinstance(x, type):
                    return None

                # If it's a subclass of KuzuRelationshipBase, it's a relationship, not a node
                if issubclass(x, KuzuRelationshipBase):
                    return None  # It's a relationship, not a node

                # If it's not a relationship, it's a node class
                return x

            from_candidates: List[Type[Any]] = []
            to_candidates: List[Type[Any]] = []
            for pair in rel_pairs:
                cf = _node_class_or_none(pair.from_node)
                ct = _node_class_or_none(pair.to_node)
                if cf is not None and cf not in from_candidates:
                    from_candidates.append(cf)
                if ct is not None and ct not in to_candidates:
                    to_candidates.append(ct)
            if (len(from_candidates) == 0) or (len(to_candidates) == 0):
                logger.error(
                    "No candidates resolved from rel_pairs for %s: from=%d to=%d rel_pairs=%r",
                    result_model_class.__name__, len(from_candidates), len(to_candidates), rel_pairs
                )

            if not rel_pairs and legacy_from is not None and legacy_to is not None:
                cf = _node_class_or_none(legacy_from)
                ct = _node_class_or_none(legacy_to)
                if cf is not None:
                    from_candidates.append(cf)
                if ct is not None:
                    to_candidates.append(ct)

            # === USE MODULE-LEVEL CACHED LOOKUPS FOR O(1) ACCESS ===
            # Get cached label->class lookup dicts (built once per relationship class)
            from_label_to_cls, to_label_to_cls = _get_rel_label_lookups(
                result_model_class, from_candidates, to_candidates
            )

            # Optimized node instance builder using module-level cached metadata
            def _build_node_instance_fast(
                node_dict: Dict[str, Any],
                label_to_cls: Dict[str, Type[Any]]
            ) -> Any:
                nonlocal cache_hits, cache_misses
                # 1) Get label (fast path)
                label = node_dict.get('_label') or node_dict.get('_LABEL')
                if label is None:
                    raise KeyError("Endpoint node missing _label")

                # 2) O(1) class lookup
                node_cls = label_to_cls.get(label)
                if node_cls is None:
                    raise KeyError(f"No candidate node class for label: {label}")

                # 3) Get pre-computed metadata from module-level cache
                meta = _get_node_class_meta(node_cls)
                pk_fields = meta['pk_fields']
                valid_fields = meta['valid_fields']
                uuid_fields = meta['uuid_fields']
                uuid_list_fields = meta['uuid_list_fields']
                optional_uuid_fields = meta.get('optional_uuid_fields', frozenset())
                fields_with_defaults = meta['fields_with_defaults']

                def _assert_uuid(value: Any, field_name: str) -> None:
                    if isinstance(value, uuid_module.UUID):
                        return
                    raise TypeError(
                        f"UUID field {field_name} in {node_cls.__name__} expected uuid.UUID, "
                        f"got {value!r} ({type(value).__name__})"
                    )

                def _assert_uuid_seq(seq_value: Any, field_name: str) -> None:
                    if not isinstance(seq_value, (list, tuple)):
                        raise TypeError(
                            f"UUID[] field {field_name} in {node_cls.__name__} expected list/tuple of uuid.UUID, "
                            f"got {seq_value!r} ({type(seq_value).__name__})"
                        )
                    for item in seq_value:
                        _assert_uuid(item, field_name)

                # 4) Build cache key from PK values with strict UUID validation
                pk_values: List[Any] = []
                for pk in pk_fields:
                    v_pk = node_dict.get(pk)
                    if v_pk is None:
                        raise ValueError(f"Endpoint node primary key field {pk} is None")
                    if pk in uuid_list_fields:
                        if isinstance(v_pk, list):
                            _assert_uuid_seq(v_pk, pk)
                            v_pk = tuple(v_pk)
                        elif isinstance(v_pk, tuple):
                            _assert_uuid_seq(v_pk, pk)
                        else:
                            _assert_uuid_seq(v_pk, pk)
                    elif pk in uuid_fields:
                        _assert_uuid(v_pk, pk)
                    pk_values.append(v_pk)

                cache_key = (label, tuple(pk_values))
                cached = endpoint_cache.get(cache_key)
                if cached is not None:
                    cache_hits += 1
                    return cached
                cache_misses += 1

                # 5) Build clean props dict with strict UUID type checks (no coercion)
                clean_props: Dict[str, Any] = {}
                for k, v in node_dict.items():
                    if k.startswith('_'):
                        continue
                    if k not in valid_fields:
                        continue
                    # Skip None for fields with defaults
                    if v is None and k in fields_with_defaults:
                        continue

                    if (k in optional_uuid_fields) and (v == _NULL_UUID):
                        v = None

                    if k in uuid_fields:
                        if v is None and k in optional_uuid_fields:
                            clean_props[k] = None
                            continue
                        _assert_uuid(v, k)
                    elif k in uuid_list_fields:
                        if v is None and k in optional_uuid_fields:
                            clean_props[k] = None
                            continue
                        _assert_uuid_seq(v, k)

                    clean_props[k] = v

                # 6) Use model_construct for fast instantiation (skip validation - data from DB is valid)
                inst = node_cls.model_construct(**clean_props)
                endpoint_cache[cache_key] = inst
                return inst

            for row in raw_results:
                # Extract endpoints by standardized keys (strict, two forms)
                if ((DDLConstants.REL_FROM_NODE_FIELD in row) and
                    (DDLConstants.REL_TO_NODE_FIELD in row)):
                    from_data = row[DDLConstants.REL_FROM_NODE_FIELD]
                    to_data = row[DDLConstants.REL_TO_NODE_FIELD]
                elif ((QueryReturnAliasConstants.FROM_ENDPOINT in row) and
                      (QueryReturnAliasConstants.TO_ENDPOINT in row)):
                    from_data = row[QueryReturnAliasConstants.FROM_ENDPOINT]
                    to_data = row[QueryReturnAliasConstants.TO_ENDPOINT]
                else:
                    logger.error(
                        "Missing required endpoint aliases in row; keys=%r",
                        list(row.keys()),
                    )
                    raise KeyError("Relationship query results missing required endpoint aliases")

                # Normalize endpoint maps to ensure lowercase '_label' exists when only uppercase is present
                if isinstance(from_data, dict) and ('_label' not in from_data) and ('_LABEL' in from_data):
                    from_data['_label'] = from_data['_LABEL']
                if isinstance(to_data, dict) and ('_label' not in to_data) and ('_LABEL' in to_data):
                    to_data['_label'] = to_data['_LABEL']

                # Validate endpoint presence and structure (strict)
                if type(from_data) is not dict or type(to_data) is not dict:
                    logger.error(
                        "Endpoint nodes are not dicts: from=%r to=%r",
                        type(from_data), type(to_data)
                    )
                    raise TypeError("Endpoint nodes must be dictionaries with Kuzu metadata")
                from_label_present = (('_label' in from_data) or ('_LABEL' in from_data))
                to_label_present = (('_label' in to_data) or ('_LABEL' in to_data))
                if not from_label_present or not to_label_present:
                    logger.error(
                        "Endpoint node missing _label: from_keys=%r to_keys=%r",
                        list(from_data.keys()), list(to_data.keys())
                    )
                    raise KeyError("Endpoint nodes missing _label")

                from_node = _build_node_instance_fast(from_data, from_label_to_cls)
                to_node = _build_node_instance_fast(to_data, to_label_to_cls)
                # from_node and to_node must be constructed; any failure would have raised earlier

                # Collect relationship properties: prefer nested alias dict when returned
                # Otherwise parse qualified fields from the row
                # Relationship properties must be present under the result alias (no fallbacks)
                if result_alias not in row:
                    logger.error(
                        "Missing relationship alias %r in row; keys=%r",
                        result_alias,
                        list(row.keys()),
                    )
                    raise KeyError("Relationship properties missing for alias")
                rel_dict = row[result_alias]
                if type(rel_dict) is not dict:
                    logger.error(
                        "Relationship alias %r value is not dict: %r",
                        result_alias,
                        type(rel_dict),
                    )
                    raise TypeError("Relationship alias value must be a dictionary")
                rel_props: Dict[str, Any] = {
                    k: v for (k, v) in rel_dict.items() if not str(k).startswith('_')
                }

                # Ensure structural endpoint fields are not included as properties
                if DDLConstants.REL_FROM_NODE_FIELD in rel_props:
                    rel_props.pop(DDLConstants.REL_FROM_NODE_FIELD)
                if DDLConstants.REL_TO_NODE_FIELD in rel_props:
                    rel_props.pop(DDLConstants.REL_TO_NODE_FIELD)

                # Construct relationship instance using model_construct for speed (data from DB is valid)
                instance = result_model_class.model_construct(from_node=from_node, to_node=to_node, **rel_props)
                mapped.append(instance)

            t_map_end = time.perf_counter()
            if getattr(self._session, "_debug_timing", False) or ((t_map_end - t_map_start) >= 0.25):
                logger.info(
                    "kuzu.query.map_results rel=%s rows=%d endpoints_unique=%d cache_hits=%d cache_misses=%d seconds=%.6f",
                    result_model_class.__name__,
                    len(raw_results),
                    len(endpoint_cache),
                    int(cache_hits),
                    int(cache_misses),
                    (t_map_end - t_map_start),
                )

            return mapped

        mapped = []
        for row in raw_results:
            if self._state.select_fields:
                # Partial model with selected fields
                instance_data = {}
                for field in self._state.select_fields:
                    if "." in field:
                        alias, field_name = field.split(".", 1)
                        if alias in row and field_name in row[alias]:
                            instance_data[field_name] = row[alias][field_name]
                    elif field in row:
                        instance_data[field] = row[field]
                    elif result_alias in row and field in row[result_alias]:
                        instance_data[field] = row[result_alias][field]
                    else:
                        qualified = f"{result_alias}.{field}"
                        if qualified in row:
                            instance_data[field] = row[qualified]

                # Create partial instance (node queries only)
                instance = result_model_class.model_construct(**instance_data)
                mapped.append(instance)
            else:
                # Full model instance
                if result_alias in row:
                    node_data = row[result_alias]
                    # Filter out Kuzu internal fields
                    cleaned_data = {k: v for k, v in node_data.items()
                                   if not k.startswith('_')}
                    instance = result_model_class(**cleaned_data)
                    mapped.append(instance)
                elif len(row) == 1 and type(list(row.values())[0]) is dict:
                    # Single node result
                    node_data = list(row.values())[0]
                    # Filter out Kuzu internal fields
                    cleaned_data = {k: v for k, v in node_data.items()
                                   if not k.startswith('_')}
                    instance = result_model_class(**cleaned_data)
                    mapped.append(instance)
                else:
                    # @@ STEP: Handle UNION query results with qualified field names
                    # || S.1: Check if this is a UNION query result with qualified field names
                    qualified_fields = {}
                    for key, value in row.items():
                        if "." in key:
                            alias, field_name = key.split(".", 1)
                            if alias == result_alias:
                                qualified_fields[field_name] = value

                    if qualified_fields:
                        # || S.2: Create instance from qualified fields
                        instance = result_model_class(**qualified_fields)
                        mapped.append(instance)
                    else:
                        # || S.3: Raw dict result (fallback)
                        mapped.append(row)

        return mapped

    def __iter__(self) -> Iterator[Union[ModelType, Dict[str, Any]]]:
        """Iterate over query results."""
        return iter(self.all())

    def __repr__(self) -> str:
        """String representation of the query."""
        cypher, _ = self.to_cypher()
        return f"<Query({self._state.model_class.__name__}): {cypher[:100]}...>"

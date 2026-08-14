from __future__ import annotations
from typing import TYPE_CHECKING, Any, Generic, Iterator, Type, TypeVar
from atp_pipeline import normalize_join_type, normalize_relationship_direction
from .constants import RelationshipDirection, ValidationMessageConstants
from .kuzu_query_builder import CypherQueryBuilder, JoinClause, QueryState
from .kuzu_query_expressions import AggregateFunction, FilterExpression, JoinType, OrderDirection
from .kuzu_query_fields import ModelFieldAccessor, QueryField
from .kuzu_relationship_read import (
    can_use_native_relationship_read,
    construct_model_from_db_payload,
    materialize_endpoint_node,
    model_payload_fields,
    relationship_read_filter_statements,
    relationship_endpoint_types,
)
if TYPE_CHECKING:
    from .kuzu_session import KuzuSession
ModelType = TypeVar("ModelType")
class Query(Generic[ModelType]):
    """Build an ORM query AST and submit it through the session gateway."""
    def __init__(
        self,
        model_class: Type[ModelType],
        session: KuzuSession | None = None,
        alias: str = "n",
    ) -> None:
        self._state = QueryState(model_class=model_class, alias=alias)
        self._session = session
        self._fields = ModelFieldAccessor(model_class)
    @property
    def fields(self) -> ModelFieldAccessor:
        return self._fields
    def _copy_with_state(self, **kwargs: Any) -> Query[ModelType]:
        obj = Query.__new__(Query)
        obj._state = self._state.copy(**kwargs)
        obj._session = self._session
        obj._fields = self._fields
        return obj
    def filter(self, *expressions: FilterExpression) -> Query[ModelType]:
        return self._copy_with_state(filters=[*self._state.filters, *expressions])
    def where(self, expression: FilterExpression) -> Query[ModelType]:
        return self.filter(expression)
    def filter_by(self, **kwargs: Any) -> Query[ModelType]:
        target = self._state.return_model_class or self._state.model_class
        return self.filter(*(QueryField(name, target) == value for name, value in kwargs.items()))
    def order_by(self, *fields: str | tuple[str, OrderDirection] | QueryField) -> Query[ModelType]:
        items = list(self._state.order_by)
        for field in fields:
            if isinstance(field, str):
                items.append((field, OrderDirection.ASC))
            elif isinstance(field, tuple):
                items.append(field)
            elif isinstance(field, QueryField):
                items.append((field.field_name, OrderDirection.ASC))
            else:
                raise ValueError(ValidationMessageConstants.INVALID_ORDER_BY_ARGUMENT.format(field))
        return self._copy_with_state(order_by=items)
    def limit(self, count: int) -> Query[ModelType]:
        return self._copy_with_state(limit_value=count)
    def offset(self, count: int) -> Query[ModelType]:
        return self._copy_with_state(offset_value=count)
    def distinct(self) -> Query[ModelType]:
        return self._copy_with_state(distinct=True)
    def select(self, *fields: str | QueryField) -> Query[ModelType]:
        names: list[str] = []
        for field in fields:
            if isinstance(field, str):
                names.append(field)
            elif isinstance(field, QueryField):
                names.append(field.field_name)
            else:
                raise ValueError(ValidationMessageConstants.INVALID_SELECT_FIELD.format(field))
        return self._copy_with_state(select_fields=names)
    def pairs_subset(self, indices: list[int]) -> Query[ModelType]:
        if not isinstance(indices, list) or any(type(index) is not int or index < 0 for index in indices):
            raise ValueError("pairs_subset expects non-negative integer indices")
        return self._copy_with_state(pairs_subset=list(indices))
    def return_raw(self) -> Query[ModelType]:
        return self._copy_with_state(return_raw=True)
    def join(
        self,
        target_model_or_rel: Type[Any],
        condition_or_model: Any | None = None,
        join_type: JoinType | str = JoinType.INNER,
        target_alias: str | None = None,
        rel_alias: str | None = None,
        conditions: list[FilterExpression] | None = None,
        direction: Any | None = None,
        pattern: str | None = None,
        properties: dict[str, Any] | None = None,
        min_hops: int = 1,
        max_hops: int = 1,
    ) -> Query[ModelType]:
        join_type = _normalize_join_type(join_type)
        is_rel = isinstance(target_model_or_rel, type) and "__kuzu_rel_name__" in target_model_or_rel.__dict__
        relationship_class = target_model_or_rel if is_rel else condition_or_model
        target_model = condition_or_model if is_rel and isinstance(condition_or_model, type) else target_model_or_rel
        if is_rel and not isinstance(condition_or_model, type):
            conditions = [condition_or_model] if condition_or_model is not None else conditions
            target_model = None
        if target_alias is None and isinstance(target_model, type):
            target_alias = f"{target_model.__name__.lower()}_joined"
        clause = JoinClause(
            relationship_class=relationship_class if isinstance(relationship_class, type) else None,
            target_model=target_model if isinstance(target_model, type) else None,
            join_type=join_type,
            source_alias=self._state.alias,
            target_alias=target_alias,
            rel_alias=rel_alias,
            conditions=list(conditions or []),
            direction=_normalize_relationship_direction(direction),
            pattern=pattern,
            properties=dict(properties or {}),
            min_hops=min_hops,
            max_hops=max_hops,
        )
        return self._copy_with_state(joins=[*self._state.joins, clause])
    def outerjoin(self, target_model_or_rel: Type[Any], *args: Any, **kwargs: Any) -> Query[ModelType]:
        kwargs["join_type"] = JoinType.OPTIONAL
        return self.join(target_model_or_rel, *args, **kwargs)
    def traverse(
        self,
        relationship_class: Type[Any],
        target_model: Type[Any] | None = None,
        *,
        direction: Any = None,
        conditions: list[FilterExpression] | None = None,
        rel_alias: str | None = None,
        target_alias: str | None = None,
        min_hops: int = 1,
        max_hops: int = 1,
    ) -> Query[ModelType]:
        pairs = getattr(relationship_class, "__kuzu_relationship_pairs__", [])
        if not pairs:
            raise ValueError(f"{relationship_class.__name__} has no relationship pairs")
        normalized_direction = _normalize_relationship_direction(direction)
        if target_model is None:
            target_name = (
                pairs[0].get_from_name()
                if normalized_direction == RelationshipDirection.INCOMING
                else pairs[0].get_to_name()
            )
            from .kuzu_orm import get_node_by_name

            target_model = get_node_by_name(target_name)
        if target_model is None:
            raise ValueError(f"Traversal target is not registered: {target_name}")
        return self.join(
            relationship_class,
            target_model,
            direction=normalized_direction,
            conditions=conditions,
            rel_alias=rel_alias,
            target_alias=target_alias,
            min_hops=min_hops,
            max_hops=max_hops,
        )._copy_with_state(
            return_model_class=target_model,
            return_alias=target_alias or f"{target_model.__name__.lower()}_joined",
        )
    def outgoing(
        self,
        relationship_class: Type[Any],
        target_model: Type[Any] | None = None,
        **kwargs: Any,
    ) -> Query[ModelType]:
        return self.traverse(
            relationship_class,
            target_model,
            direction=RelationshipDirection.OUTGOING,
            **kwargs,
        )
    def incoming(
        self,
        relationship_class: Type[Any],
        target_model: Type[Any] | None = None,
        **kwargs: Any,
    ) -> Query[ModelType]:
        return self.traverse(
            relationship_class,
            target_model,
            direction=RelationshipDirection.INCOMING,
            **kwargs,
        )
    def related(
        self,
        relationship_class: Type[Any],
        target_model: Type[Any] | None = None,
        **kwargs: Any,
    ) -> Query[ModelType]:
        return self.traverse(
            relationship_class,
            target_model,
            direction=RelationshipDirection.BOTH,
            **kwargs,
        )
    def group_by(self, *fields: str) -> Query[ModelType]:
        return self._copy_with_state(group_by=list(fields))
    def having(self, expression: FilterExpression) -> Query[ModelType]:
        return self._copy_with_state(having=expression)
    def aggregate(self, alias: str, func: AggregateFunction, field: str) -> Query[ModelType]:
        aggregations = dict(self._state.aggregations)
        aggregations[alias] = (func, field)
        return self._copy_with_state(aggregations=aggregations, return_raw=True)
    def count(self, field: str = "*", alias: str = "count") -> Query[ModelType]:
        return self.aggregate(alias, AggregateFunction.COUNT, field)
    def sum(self, field: str, alias: str | None = None) -> Query[ModelType]:
        return self.aggregate(alias or f"sum_{field}", AggregateFunction.SUM, field)
    def avg(self, field: str, alias: str | None = None) -> Query[ModelType]:
        return self.aggregate(alias or f"avg_{field}", AggregateFunction.AVG, field)
    def min(self, field: str, alias: str | None = None) -> Query[ModelType]:
        return self.aggregate(alias or f"min_{field}", AggregateFunction.MIN, field)
    def max(self, field: str, alias: str | None = None) -> Query[ModelType]:
        return self.aggregate(alias or f"max_{field}", AggregateFunction.MAX, field)
    def union(self, other: Query[Any]) -> Query[ModelType]:
        return self._copy_with_state(union_queries=[*self._state.union_queries, (other, False)])
    def union_all(self, other: Query[Any]) -> Query[ModelType]:
        return self._copy_with_state(union_queries=[*self._state.union_queries, (other, True)])
    def with_raw(self, clause: str) -> Query[ModelType]:
        return self._copy_with_state(with_clauses=[*self._state.with_clauses, clause])
    def subquery(self, alias: str, query: Query[Any]) -> Query[ModelType]:
        subqueries = dict(self._state.subqueries)
        subqueries[alias] = query
        return self._copy_with_state(subqueries=subqueries)
    def to_cypher(self) -> tuple[str, dict[str, Any]]:
        return CypherQueryBuilder(self._state).build()
    def _execute(self) -> list[Any]:
        if self._session is None:
            raise RuntimeError("query execution requires a session")
        if can_use_native_relationship_read(self._state):
            rows = self._session._execute_relationship_read_for_query_object(
                self._state.model_class,
                self._state.alias,
                self._state.pairs_subset,
                filters=relationship_read_filter_statements(self._state),
            )
            return self._materialize(rows)
        query, params = self.to_cypher()
        return self._materialize(self._session._execute_for_query_object(query, params))

    def _ordered_for_paging(self) -> Query[ModelType]:
        """Append model identity fields as stable paging tie-breakers."""
        state = self._state
        if state.limit_value is not None or state.offset_value is not None:
            raise ValueError(
                "page_size cannot be combined with query limit or offset"
            )
        if (
            state.aggregations
            or state.group_by
            or state.distinct
            or state.union_queries
            or state.subqueries
            or state.with_clauses
            or state.joins
            or state.return_model_class is not None
            or hasattr(state.model_class, "__kuzu_rel_name__")
        ):
            raise ValueError(
                "paged query shape has no derivable total row identity; use eager execution"
            )

        order_by = list(state.order_by)
        normalized_fields = {
            field if "." in field else f"{state.alias}.{field}"
            for field, _direction in order_by
        }

        def _append_model_identity(model_class: Type[Any], alias: str) -> None:
            getter = getattr(model_class, "get_primary_key_fields", None)
            if not callable(getter):
                raise ValueError(
                    f"paged query model has no primary key metadata: {model_class.__name__}"
                )
            primary_keys = getter()
            if not isinstance(primary_keys, list) or not primary_keys:
                raise ValueError(
                    f"paged query model has empty primary key metadata: {model_class.__name__}"
                )
            if not all(isinstance(field, str) and field for field in primary_keys):
                raise ValueError(
                    f"paged query model has invalid primary key metadata: {model_class.__name__}"
                )
            for primary_key in primary_keys:
                qualified = f"{alias}.{primary_key}"
                if qualified not in normalized_fields:
                    order_by.append((qualified, OrderDirection.ASC))
                    normalized_fields.add(qualified)

        _append_model_identity(state.model_class, state.alias)
        return self._copy_with_state(order_by=order_by)

    def iter(self, page_size: int | None = None, prefetch_pages: int = 1) -> Iterator[Any]:
        if page_size is None:
            return iter(self._execute())
        if type(page_size) is not int or page_size <= 0:
            raise ValueError("page_size must be a positive integer")
        if self._session is None:
            raise RuntimeError("query execution requires a session")
        if can_use_native_relationship_read(self._state):
            endpoint_types = self._relationship_endpoint_types()
            rows = self._session._iterate_relationship_read_for_query_object(
                self._state.model_class,
                self._state.alias,
                self._state.pairs_subset,
                relationship_read_filter_statements(self._state),
                page_size,
                prefetch_pages,
            )
            return (self._materialize_row(row, endpoint_types) for row in rows)
        paged_query = self._ordered_for_paging()
        query, params = paged_query.to_cypher()
        total_rows = paged_query._copy_with_state(order_by=[]).count_results()
        endpoint_types = paged_query._relationship_endpoint_types()
        return (
            paged_query._materialize_row(row, endpoint_types)
            for row in self._session._iterate_for_query_object(
                query,
                params,
                page_size,
                prefetch_pages,
                total_rows,
            )
        )
    def all(
        self,
        *,
        as_iterator: bool = False,
        page_size: int | None = None,
        prefetch_pages: int = 1,
    ) -> list[Any] | Iterator[Any]:
        if as_iterator:
            return self.iter(page_size=10 if page_size is None else page_size, prefetch_pages=prefetch_pages)
        return list(self.iter())
    def first(self) -> Any | None:
        return (rows[0] if (rows := self.limit(1).all()) else None)
    def one(self) -> Any:
        rows = self.limit(2).all()
        if len(rows) != 1:
            raise ValueError(f"Expected exactly one result, got {len(rows)}")
        return rows[0]
    def one_or_none(self) -> Any | None:
        rows = self.limit(2).all()
        if len(rows) > 1:
            raise ValueError(f"Expected one or no results, got {len(rows)}")
        return rows[0] if rows else None
    def exists(self) -> bool: return self.limit(1).first() is not None
    def count_results(self) -> int:
        rows = self.count("*", alias="count").all()
        if not rows:
            return 0
        value = rows[0].get("count")
        if not isinstance(value, int):
            raise TypeError("count result must be an integer")
        return value
    def _materialize(self, rows: list[dict[str, Any]]) -> list[Any]:
        endpoint_types = self._relationship_endpoint_types()
        return [self._materialize_row(row, endpoint_types) for row in rows]
    def _materialize_row(
        self,
        row: dict[str, Any],
        endpoint_types: dict[str, Type[Any]] | None = None,
    ) -> Any:
        if self._state.return_raw or self._state.aggregations:
            return row
        if self._state.select_fields:
            return self._materialize_select_row(row)
        return self._materialize_page(row, endpoint_types)
    def _materialize_select_row(self, row: dict[str, Any]) -> Any:
        model_class = self._state.return_model_class or self._state.model_class
        alias = self._state.return_alias or self._state.alias
        payload: dict[str, Any] = {}
        for field in self._state.select_fields or []:
            output_name = field.rsplit(".", 1)[-1]
            for key in (field, output_name, f"{alias}.{output_name}"):
                if key in row:
                    payload[output_name] = row[key]
                    break
        return construct_model_from_db_payload(model_class, payload)
    def _materialize_page(
        self,
        row: dict[str, Any],
        endpoint_types: dict[str, Type[Any]] | None = None,
    ) -> Any:
        model_class = self._state.return_model_class or self._state.model_class
        alias = self._state.return_alias or self._state.alias
        payload = row.get(alias)
        if payload is None and len(row) == 1:
            payload = next(iter(row.values()))
        if not isinstance(payload, dict):
            raise TypeError("ORM materialization requires a dictionary payload")
        if hasattr(model_class, "__kuzu_rel_name__"):
            if endpoint_types is None:
                endpoint_types = relationship_endpoint_types(model_class)
            payload = dict(payload)
            payload["from_node"] = materialize_endpoint_node(row.get("from_node"), endpoint_types)
            payload["to_node"] = materialize_endpoint_node(row.get("to_node"), endpoint_types)
        return construct_model_from_db_payload(model_class, payload)
    def _relationship_endpoint_types(self) -> dict[str, Type[Any]] | None:
        model_class = self._state.return_model_class or self._state.model_class
        if hasattr(model_class, "__kuzu_rel_name__"):
            return relationship_endpoint_types(model_class)
        return None
    def __iter__(self) -> Iterator[Any]: return self.iter()
    def __repr__(self) -> str:
        query = self.to_cypher()[0]
        return f"Query({self._state.model_class.__name__}, {query!r})"
def _model_payload(model_class: Type[Any], payload: dict[str, Any]) -> dict[str, Any]:
    fields = model_payload_fields(model_class)
    return {key: value for key, value in payload.items() if key in fields}

def _normalize_join_type(join_type: JoinType | str) -> JoinType:
    if isinstance(join_type, JoinType):
        return join_type
    if not isinstance(join_type, str):
        raise TypeError("join_type must be a JoinType or string")
    return JoinType(normalize_join_type(join_type))

def _normalize_relationship_direction(direction: Any) -> Any:
    if direction is None:
        return None
    if not isinstance(direction, str):
        return direction
    return normalize_relationship_direction(direction)

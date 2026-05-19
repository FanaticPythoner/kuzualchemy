from __future__ import annotations
from typing import TYPE_CHECKING, Any, Generic, Iterator, Type, TypeVar
from .constants import ValidationMessageConstants
from .kuzu_query_builder import CypherQueryBuilder, JoinClause, QueryState
from .kuzu_query_expressions import AggregateFunction, FilterExpression, JoinType, OrderDirection
from .kuzu_query_fields import ModelFieldAccessor, QueryField
from .kuzu_relationship_read import can_use_native_relationship_read, materialize_endpoint_node
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
        if isinstance(join_type, str):
            join_type = JoinType(join_type.lower())
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
            direction=direction,
            pattern=pattern,
            properties=dict(properties or {}),
            min_hops=min_hops,
            max_hops=max_hops,
        )
        return self._copy_with_state(joins=[*self._state.joins, clause])
    def outerjoin(self, target_model_or_rel: Type[Any], *args: Any, **kwargs: Any) -> Query[ModelType]:
        kwargs["join_type"] = JoinType.OPTIONAL
        return self.join(target_model_or_rel, *args, **kwargs)
    def traverse(self, relationship_class: Type[Any], direction: Any = None) -> Query[ModelType]:
        pairs = getattr(relationship_class, "__kuzu_relationship_pairs__", [])
        if not pairs:
            raise ValueError(f"{relationship_class.__name__} has no relationship pairs")
        target_name = pairs[0].get_to_name()
        from .kuzu_orm import get_node_by_name
        target_model = get_node_by_name(target_name)
        if target_model is None:
            raise ValueError(f"Traversal target is not registered: {target_name}")
        return self.join(relationship_class, target_model, direction=direction)._copy_with_state(
            return_model_class=target_model,
            return_alias=f"{target_model.__name__.lower()}_joined",
        )
    def outgoing(self, relationship_class: Type[Any]) -> Query[ModelType]:
        return self.traverse(relationship_class)
    def incoming(self, relationship_class: Type[Any]) -> Query[ModelType]:
        return self.traverse(relationship_class)
    def related(self, relationship_class: Type[Any]) -> Query[ModelType]:
        return self.traverse(relationship_class)
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
            )
            return self._materialize(rows)
        query, params = self.to_cypher()
        return self._materialize(self._session._execute_for_query_object(query, params))
    def iter(self, page_size: int | None = None, prefetch_pages: int = 1) -> Iterator[Any]:
        return iter(self._execute())
    def all(self) -> list[Any]:
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
    def count_results(self) -> int: return len(self.all())
    def _materialize(self, rows: list[dict[str, Any]]) -> list[Any]:
        if self._state.return_raw or self._state.select_fields or self._state.aggregations:
            return rows
        model_class = self._state.return_model_class or self._state.model_class
        alias = self._state.return_alias or self._state.alias
        values: list[Any] = []
        for row in rows:
            payload = row.get(alias)
            if payload is None and len(row) == 1:
                payload = next(iter(row.values()))
            if not isinstance(payload, dict):
                raise TypeError("ORM materialization requires a dictionary payload")
            if hasattr(model_class, "__kuzu_rel_name__"):
                payload = dict(payload)
                payload["from_node"] = materialize_endpoint_node(row.get("from_node"))
                payload["to_node"] = materialize_endpoint_node(row.get("to_node"))
            values.append(model_class(**_model_payload(model_class, payload)))
        return values
    def __iter__(self) -> Iterator[Any]: return self.iter()
    def __repr__(self) -> str:
        query, _ = self.to_cypher()
        return f"Query({self._state.model_class.__name__}, {query!r})"
def _model_payload(model_class: Type[Any], payload: dict[str, Any]) -> dict[str, Any]:
    fields = getattr(model_class, "model_fields", None)
    if isinstance(fields, dict):
        return {key: value for key, value in payload.items() if key in fields}
    raise TypeError(f"{model_class.__name__} has no Pydantic field map")

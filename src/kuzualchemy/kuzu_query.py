"""
Main Query class with method chaining for Kuzu ORM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union, TypeVar, Tuple, Iterator, Generic

from .constants import ValidationMessageConstants
from .kuzu_query_expressions import (
    FilterExpression, AggregateFunction, OrderDirection, JoinType,
)
from .kuzu_query_builder import QueryState, JoinClause, CypherQueryBuilder
from .kuzu_query_fields import QueryField, ModelFieldAccessor

if TYPE_CHECKING:
    from .kuzu_session import KuzuSession

ModelType = TypeVar("ModelType")


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
        target_alias = self._state.return_alias or self._state.alias

        for field_name, value in kwargs.items():
            # Create field with explicit alias for traversal context
            if self._state.return_alias:
                # In traversal context, use qualified field path
                field_path = f"{target_alias}.{field_name}"
                field = QueryField(field_path, target_model)
            else:
                # Normal context, use unqualified field name
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
    
    def join(
        self,
        target_model_or_rel: Type[Any],
        condition_or_model: Optional[Any] = None,
        join_type: Union[JoinType, str] = JoinType.INNER,
        target_alias: Optional[str] = None,
        rel_alias: Optional[str] = None,
        conditions: Optional[List[FilterExpression]] = None,
        **kwargs
    ) -> Query:
        """Join with another node through a relationship.

        Supports two calling patterns:
        1. join(TargetModel, relationship_class, ...)
        2. join(RelationshipClass, condition, ...)
        """
        # @@ STEP: Handle different join calling patterns
        # || S.S.1: Check if first arg is a relationship class
        if hasattr(target_model_or_rel, '__kuzu_rel_name__'):
            # Pattern 2: join(RelationshipClass, condition, ...)
            relationship_class = target_model_or_rel
            target_model = None
            if condition_or_model and not hasattr(condition_or_model, '__kuzu_node_name__'):
                # It's a condition
                conditions = [condition_or_model] if condition_or_model else []
            else:
                # It's a model
                target_model = condition_or_model
                conditions = conditions or []
        else:
            # Pattern 1: join(TargetModel, relationship_class, ...)
            target_model = target_model_or_rel
            relationship_class = condition_or_model if hasattr(condition_or_model or {}, '__kuzu_rel_name__') else None
            conditions = conditions or []

        # || S.S.2: Convert string join_type to enum
        if isinstance(join_type, str):
            join_type = JoinType[join_type.upper()] if join_type.upper() in JoinType.__members__ else JoinType.INNER

        if target_model and not target_alias:
            target_alias = f"{target_model.__name__.lower()}_joined"

        join_clause = JoinClause(
            relationship_class=relationship_class,
            target_model=target_model,
            join_type=join_type,
            source_alias=self._state.alias,
            target_alias=target_alias,
            rel_alias=rel_alias,
            conditions=conditions,
            **kwargs
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
        return self._session.execute(cypher, params)
    
    def all(self) -> Union[List[ModelType], List[Dict[str, Any]]]:
        """Execute query and return all results."""
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
        count_query = self.count()
        result = count_query._execute()
        if result and len(result) > 0:
            return result[0].get("count", 0)
        return 0
    
    def _map_results(self, raw_results: List[Dict[str, Any]]) -> Union[List[ModelType], List[Dict[str, Any]]]:
        """Map raw results to model instances or return raw dictionaries for special cases."""
        if self._state.return_raw:
            return raw_results

        # @@ STEP: For GROUP BY queries with aggregations, return raw dictionaries
        # || S.1: GROUP BY queries return grouped fields + aggregated values, not full model instances
        if self._state.aggregations:
            return raw_results

        # @@ STEP: Determine which alias and model class to use for result mapping
        result_alias = self._state.return_alias or self._state.alias
        result_model_class = self._state.return_model_class or self._state.model_class

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

                # Create partial instance
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
                elif len(row) == 1 and isinstance(list(row.values())[0], dict):
                    # Single node result
                    node_data = list(row.values())[0]
                    # Filter out Kuzu internal fields
                    cleaned_data = {k: v for k, v in node_data.items()
                                   if not k.startswith('_')}
                    instance = result_model_class(**cleaned_data)
                    mapped.append(instance)
                else:
                    # @@ STEP: Handle UNION query results with qualified field names
                    # || S.1: Check if this is a UNION query result with qualified field names (e.g., "n.field_name")
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

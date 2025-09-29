# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
Main Query class with method chaining for Kuzu ORM.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING, Any, Dict, List, Optional, Type, Union, TypeVar, Tuple, Iterator, Generic
)
import logging

from .constants import ValidationMessageConstants, DDLConstants, QueryReturnAliasConstants, KuzuDataType
from .kuzu_query_expressions import (
    FilterExpression, AggregateFunction, OrderDirection, JoinType,
)
from .kuzu_query_builder import QueryState, JoinClause, CypherQueryBuilder
from .kuzu_query_fields import QueryField, ModelFieldAccessor

logger = logging.getLogger(__name__)

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

            # Build candidate classes from relationship pairs (handles multi-pair and legacy)
            has_pairs_attr = '__kuzu_relationship_pairs__' in d_model
            rel_pairs = d_model['__kuzu_relationship_pairs__'] if has_pairs_attr else []
            legacy_from = d_model['__kuzu_from_node__'] if '__kuzu_from_node__' in d_model else None
            legacy_to = d_model['__kuzu_to_node__'] if '__kuzu_to_node__' in d_model else None
            logger.debug(
                "RelClass=%s has_pairs_attr=%s pairs_len=%s legacy_from=%s legacy_to=%s",
                result_model_class.__name__, has_pairs_attr, len(rel_pairs),
                type(legacy_from).__name__ if legacy_from is not None else None,
                type(legacy_to).__name__ if legacy_to is not None else None,
            )
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
                try:
                    if issubclass(x, KuzuRelationshipBase):
                        return None  # It's a relationship, not a node
                except TypeError:
                    # issubclass can raise TypeError if x is not a class
                    return None

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

            # Helper to build a validated node instance from endpoint data (strict)
            def _build_node_instance(
                node_dict: Dict[str, Any],
                candidates: List[Type[Any]]
            ) -> Any:
                # 1) Strictly require dict with _label and a candidate class match
                if type(node_dict) is not dict:
                    logger.error("Endpoint data is not a dict: %r", type(node_dict))
                    raise TypeError("Endpoint data must be a dict")
                if '_label' not in node_dict:
                    logger.error("Endpoint dict missing _label; keys: %r", list(node_dict.keys()))
                    raise KeyError("Endpoint node missing _label")
                label = node_dict['_label']
                # 2) Resolve class by label strictly among declared candidates
                node_cls: Optional[Type[Any]] = None
                for cls in candidates:
                    d = cls.__dict__
                    cand_name = (
                        d['__kuzu_node_name__'] if ('__kuzu_node_name__' in d)
                        else cls.__name__
                    )
                    if cand_name == label:
                        node_cls = cls
                        break
                if node_cls is None:
                    cand_labels = [
                        (c.__dict__['__kuzu_node_name__']
                         if ('__kuzu_node_name__' in c.__dict__)
                         else c.__name__)
                        for c in candidates
                    ]
                    logger.error(
                        "No candidate class found for label %r among %r",
                        label, cand_labels
                    )
                    raise KeyError(f"No candidate node class for label: {label}")

                # 3) Filter properties to only include fields that belong to this node class
                # Kuzu returns ALL possible fields from ALL node types, with None for non-matching
                # We must exclude None values for fields that don't belong to this specific class
                valid_fields = set(node_cls.model_fields.keys())
                clean_props = {}
                for k, v in node_dict.items():
                    if not str(k).startswith('_'):
                        if k in valid_fields:
                            # Include field if it belongs to this node class and has non-None value
                            # OR if it's None but field doesn't have default (making None valid)
                            fi = node_cls.model_fields[k]
                            has_default = (
                                (fi.default is not None) or
                                (fi.default_factory is not None)
                            )

                            if v is not None or not has_default:
                                clean_props[k] = v
                            # Skip None values for fields with defaults - let Pydantic use default
                        # Exclude fields that don't belong to this node class (even if non-None)
                        # This prevents cross-contamination between different node types in results

                return node_cls(**clean_props)

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

                # Validate endpoint presence and structure (strict)
                if type(from_data) is not dict or type(to_data) is not dict:
                    logger.error(
                        "Endpoint nodes are not dicts: from=%r to=%r",
                        type(from_data), type(to_data)
                    )
                    raise TypeError("Endpoint nodes must be dictionaries with Kuzu metadata")
                if ('_label' not in from_data) or ('_label' not in to_data):
                    logger.error(
                        "Endpoint node missing _label: from_keys=%r to_keys=%r",
                        list(from_data.keys()), list(to_data.keys())
                    )
                    raise KeyError("Endpoint nodes missing _label")

                from_node = _build_node_instance(from_data, from_candidates)
                to_node = _build_node_instance(to_data, to_candidates)
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

                # Construct validated relationship instance (no model_construct bypass)
                instance = result_model_class(from_node=from_node, to_node=to_node, **rel_props)
                mapped.append(instance)

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

                    cleaned_data = self._convert_uuid_objects_to_strings(cleaned_data, result_model_class)

                    instance = result_model_class(**cleaned_data)
                    mapped.append(instance)
                elif len(row) == 1 and type(list(row.values())[0]) is dict:
                    # Single node result
                    node_data = list(row.values())[0]
                    # Filter out Kuzu internal fields
                    cleaned_data = {k: v for k, v in node_data.items()
                                   if not k.startswith('_')}

                    cleaned_data = self._convert_uuid_objects_to_strings(cleaned_data, result_model_class)

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

    def _convert_uuid_objects_to_strings(self, data: Dict[str, Any], model_class: Type[Any]) -> Dict[str, Any]:
        """
        Convert UUID objects to strings for UUID fields in the model.

        Args:
            data: Dictionary of field names to values
            model_class: The model class to check field types against

        Returns:
            Dictionary with UUID objects converted to strings for UUID fields
        """
        # @@ STEP 1: Get field metadata to identify UUID fields
        from .kuzu_orm import _kuzu_registry
        converted_data = {}
        for field_name, value in data.items():
            # || S.1.1: Get field metadata
            if hasattr(model_class, 'model_fields') and field_name in model_class.model_fields:
                field_info = model_class.model_fields[field_name]
                meta = _kuzu_registry.get_field_metadata(field_info)
                # || S.1.2: Convert UUID objects to strings for UUID fields
                if meta and meta.kuzu_type == KuzuDataType.UUID and hasattr(value, '__class__'):
                    # Check if it's a UUID object (avoid importing uuid if not needed)
                    if value.__class__.__name__ == 'UUID':
                        converted_data[field_name] = str(value)
                    else:
                        converted_data[field_name] = value
                else:
                    converted_data[field_name] = value
            else:
                converted_data[field_name] = value
        return converted_data


    def __iter__(self) -> Iterator[Union[ModelType, Dict[str, Any]]]:
        """Iterate over query results."""
        return iter(self.all())

    def __repr__(self) -> str:
        """String representation of the query."""
        cypher, _ = self.to_cypher()
        return f"<Query({self._state.model_class.__name__}): {cypher[:100]}...>"

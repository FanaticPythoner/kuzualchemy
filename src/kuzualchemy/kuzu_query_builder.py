# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
Query state management and Cypher query builder for Kuzu ORM.
"""

from __future__ import annotations
from typing import Any, Optional, Type, Dict, List, Tuple
import logging
from dataclasses import dataclass, field
from .kuzu_query_expressions import (
    FilterExpression, AggregateFunction, OrderDirection, JoinType
)
from .constants import DDLConstants, ValidationMessageConstants, JoinPatternConstants, RelationshipDirection, CypherConstants, QueryReturnAliasConstants

logger = logging.getLogger(__name__)

@dataclass
class JoinClause:
    """Represents a join operation in a query."""
    relationship_class: Optional[Type[Any]]
    target_model: Optional[Type[Any]]
    join_type: JoinType = JoinType.INNER
    source_alias: Optional[str] = None
    target_alias: Optional[str] = None
    rel_alias: Optional[str] = None
    conditions: List[FilterExpression] = field(default_factory=list)
    direction: Optional[Any] = None
    pattern: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    min_hops: int = 1
    max_hops: int = 1
    
    def to_cypher(self, source: str, alias_map: Dict[str, str]) -> str:
        """Convert join to Cypher pattern."""
        # @@ STEP: alias_map parameter reserved for future use in complex join patterns
        _ = alias_map  # Mark as intentionally unused

        if self.pattern:
            return self.pattern.format(
                source=source,
                target=self.target_alias,
                rel=self.rel_alias
            )
        
        if self.relationship_class:
            if not hasattr(self.relationship_class, '__kuzu_rel_name__'):
                raise ValueError(
                    ValidationMessageConstants.MISSING_KUZU_REL_NAME.format(self.relationship_class.__name__)
                )
            rel_name = self.relationship_class.__kuzu_rel_name__
            rel_pattern = f"{self.rel_alias}:{rel_name}" if self.rel_alias else f":{rel_name}"
        else:
            rel_pattern = self.rel_alias or ""
        
        if self.min_hops != 1 or self.max_hops != 1:
            if self.min_hops == self.max_hops:
                rel_pattern += f"{JoinPatternConstants.HOP_PREFIX}{self.min_hops}"
            else:
                rel_pattern += f"{JoinPatternConstants.HOP_PREFIX}{self.min_hops}{JoinPatternConstants.HOP_SEPARATOR}{self.max_hops}"
        
        if self.properties:
            prop_str = JoinPatternConstants.PROPERTY_SEPARATOR.join(f"{k}: {JoinPatternConstants.PROPERTY_PREFIX}{v}" for k, v in self.properties.items())
            rel_pattern += f" {{{prop_str}}}"
        
        if self.direction:
            # Handle both enum and string direction values
            if (self.direction == RelationshipDirection.OUTGOING or
                (hasattr(self.direction, 'name') and self.direction.name == 'FORWARD')):
                pattern = JoinPatternConstants.OUTGOING_PATTERN.format(source=source, rel_pattern=rel_pattern, target=self.target_alias)
            elif (self.direction == RelationshipDirection.INCOMING or
                  (hasattr(self.direction, 'name') and self.direction.name == 'BACKWARD')):
                pattern = JoinPatternConstants.INCOMING_PATTERN.format(source=source, rel_pattern=rel_pattern, target=self.target_alias)
            elif self.direction == RelationshipDirection.BOTH:
                pattern = JoinPatternConstants.BOTH_PATTERN.format(source=source, rel_pattern=rel_pattern, target=self.target_alias)
            else:
                # Default to outgoing for unknown directions
                pattern = JoinPatternConstants.OUTGOING_PATTERN.format(source=source, rel_pattern=rel_pattern, target=self.target_alias)
        else:
            pattern = JoinPatternConstants.OUTGOING_PATTERN.format(source=source, rel_pattern=rel_pattern, target=self.target_alias)
        
        if self.target_model:
            if not hasattr(self.target_model, '__kuzu_node_name__'):
                raise ValueError(
                    ValidationMessageConstants.MISSING_KUZU_NODE_NAME.format(self.target_model.__name__)
                )
            target_label = self.target_model.__kuzu_node_name__
            pattern = pattern.replace(f"({self.target_alias})", f"({self.target_alias}:{target_label})")
        
        if self.join_type == JoinType.OPTIONAL:
            pattern = f"{JoinPatternConstants.OPTIONAL_MATCH_PREFIX}{pattern}"
        elif self.join_type == JoinType.MANDATORY:
            pass
        else:
            pattern = f"{JoinPatternConstants.MATCH_PREFIX}{pattern}"
        
        return pattern


@dataclass
class QueryState:
    """Immutable state for query building."""
    model_class: Type[Any]
    filters: List[FilterExpression] = field(default_factory=list)
    order_by: List[Tuple[str, OrderDirection]] = field(default_factory=list)
    limit_value: Optional[int] = None
    offset_value: Optional[int] = None
    distinct: bool = False
    select_fields: Optional[List[str]] = None
    aggregations: Dict[str, Tuple[AggregateFunction, str]] = field(default_factory=dict)
    group_by: List[str] = field(default_factory=list)
    having: Optional[FilterExpression] = None
    joins: List[JoinClause] = field(default_factory=list)
    with_clauses: List[str] = field(default_factory=list)
    return_raw: bool = False
    alias: str = "n"
    subqueries: Dict[str, Any] = field(default_factory=dict)
    union_queries: List[Tuple[Any, bool]] = field(default_factory=list)
    parameter_prefix: str = ""
    return_alias: Optional[str] = None  # Override return alias for traversals
    return_model_class: Optional[Type[Any]] = None  # Override model class for traversals
    # Limit relationship pairs to a subset of indices for memory-safe streaming
    pairs_subset: Optional[List[int]] = None
    
    def copy(self, **kwargs) -> QueryState:
        """Create a copy with updated fields."""
        import copy
        new_state = copy.copy(self)
        for key, value in kwargs.items():
            if not hasattr(new_state, key):
                valid_fields = [attr for attr in dir(new_state) if not attr.startswith('_') and not callable(getattr(new_state, attr))]
                raise ValueError(
                    f"Cannot update non-existent field '{key}' in QueryState. "
                    f"Valid fields are: {', '.join(valid_fields)}"
                )
            if key in ('filters', 'order_by', 'group_by', 'joins', 'with_clauses'):
                value = list(value) if value else []
            elif key in ('aggregations', 'subqueries'):
                value = dict(value) if value else {}
            setattr(new_state, key, value)
        return new_state


class CypherQueryBuilder:
    """Builds Cypher queries from QueryState."""
    
    def __init__(self, state: QueryState):
        self.state = state
        self.alias_map: Dict[str, str] = {}
        self.parameters: Dict[str, Any] = {}
        self.alias_counter = 0
    
    def build(self) -> Tuple[str, Dict[str, Any]]:
        """Build Cypher query."""
        is_relationship = hasattr(self.state.model_class, '__is_kuzu_relationship__') and self.state.model_class.__is_kuzu_relationship__
        if is_relationship:
            return self._build_relationship_query()
        else:
            return self._build_node_query()
    
    def _build_node_query(self) -> Tuple[str, Dict[str, Any]]:
        """Build query for node models."""
        clauses = []
        
        if not hasattr(self.state.model_class, '__kuzu_node_name__'):
            raise ValueError(
                f"Model {self.state.model_class.__name__} is not a registered node - "
                f"missing __kuzu_node_name__ attribute"
            )
        node_name = self.state.model_class.__kuzu_node_name__
        
        self.alias_map[self.state.alias] = self.state.alias
        # Map model class name to alias for QueryField(model=...) resolution
        self.alias_map[self.state.model_class.__name__] = self.state.alias
        match_pattern = f"({self.state.alias}:{node_name})"

        if self.state.with_clauses:
            clauses.extend(self.state.with_clauses)
        
        clauses.append(f"{CypherConstants.MATCH} {match_pattern}")
        
        for join in self.state.joins:
            join_cypher = join.to_cypher(self.state.alias, self.alias_map)
            if join.target_alias:
                self.alias_map[join.target_alias] = join.target_alias
                if join.target_model is not None:
                    # Map target model class name to its alias
                    self.alias_map[join.target_model.__name__] = join.target_alias
            if join.rel_alias:
                self.alias_map[join.rel_alias] = join.rel_alias
                if join.relationship_class is not None:
                    # Map relationship class name to its alias
                    self.alias_map[join.relationship_class.__name__] = join.rel_alias
            if not join_cypher.startswith((CypherConstants.MATCH, "OPTIONAL")):
                clauses.append(f"{CypherConstants.MATCH} {join_cypher}")
            else:
                clauses.append(join_cypher)

            self.state.filters.extend(join.conditions)

        where_clause = self._build_where_clause()
        if where_clause:
            clauses.append(where_clause)
        
        if self.state.aggregations or self.state.group_by:
            # @@ STEP: Implement HAVING using WITH clause + WHERE pattern
            if self.state.having:
                # Use WITH clause for aggregations, then WHERE for HAVING condition
                with_items = self._build_aggregation_return()
                if self.state.group_by:
                    # Kuzu requires all expressions in WITH to be aliased
                    group_fields = [f"{self.state.alias}.{f} AS {f}" for f in self.state.group_by]
                    with_items = group_fields + with_items

                clauses.append(f"{CypherConstants.WITH} {', '.join(with_items)}")

                # Add HAVING condition as WHERE clause (post-WITH context)
                having_cypher = self.state.having.to_cypher(self.alias_map, self.state.parameter_prefix, post_with=True)
                having_params = self.state.having.get_parameters()
                for key, value in having_params.items():
                    param_key = f"{self.state.parameter_prefix}{key}"
                    self.parameters[param_key] = value
                clauses.append(f"{CypherConstants.WHERE} {having_cypher}")

                # Final RETURN with same items (aliases are now available)
                final_return_items = []
                if self.state.group_by:
                    for fld in self.state.group_by:
                        # Use the field name as alias (Kuzu requires aliases in WITH)
                        final_return_items.append(fld)

                for alias, (func, fld) in self.state.aggregations.items():
                    _ = func, fld  # Mark as intentionally unused - only alias is needed in RETURN
                    final_return_items.append(alias)

                clauses.append(f"{CypherConstants.RETURN} {', '.join(final_return_items)}")
            else:
                # Standard aggregation without HAVING
                return_items = self._build_aggregation_return()
                if self.state.group_by:
                    group_fields = [f"{self.state.alias}.{f}" for f in self.state.group_by]
                    return_items = group_fields + return_items

                # @@ STEP: Validate proper GROUP BY semantics for user control
                # || S.S: Even though Kuzu uses implicit grouping, enforce explicit GROUP BY
                self._validate_group_by_semantics(return_items)

                clauses.append(f"{CypherConstants.RETURN} {', '.join(return_items)}")
        else:
            return_clause = self._build_return_clause()
            clauses.append(return_clause)
        
        if self.state.order_by:
            order_items = []
            for fld, direction in self.state.order_by:
                if "." in fld:
                    order_items.append(f"{fld} {direction.value}")
                else:
                    # @@ STEP: When we have aggregations, ORDER BY should reference aliases, not table fields
                    if self.state.aggregations and fld in self.state.aggregations:
                        order_items.append(f"{fld} {direction.value}")
                    else:
                        order_items.append(f"{self.state.alias}.{fld} {direction.value}")
            clauses.append(f"{CypherConstants.ORDER_BY} {', '.join(order_items)}")
        
        # @@ STEP: Kuzu requires SKIP instead of OFFSET, and SKIP must come before LIMIT
        # || S.1: Add SKIP and LIMIT as separate clauses in correct order for Kuzu
        if self.state.offset_value is not None:
            clauses.append(f"{CypherConstants.SKIP} {self.state.offset_value}")
        if self.state.limit_value is not None:
            clauses.append(f"{CypherConstants.LIMIT} {self.state.limit_value}")
        
        query = "\n".join(clauses)
        # @@ STEP: Prune parameters that are not referenced in the Cypher text
        if self.parameters:
            import re as _re
            used = set(_re.findall(r"\$([A-Za-z_][A-Za-z0-9_]*)", query))
            if used:
                self.parameters = {k: v for k, v in self.parameters.items() if k in used}
        if self.state.union_queries:
            union_parts = [query]
            for union_query, use_all in self.state.union_queries:
                union_cypher, union_params = union_query.to_cypher()
                union_type = CypherConstants.UNION_ALL if use_all else CypherConstants.UNION
                union_parts.append(f"{union_type}\n{union_cypher}")
                self.parameters.update(union_params)
            query = "\n".join(union_parts)
        
        return query, self.parameters
    
    def _build_relationship_query(self) -> Tuple[str, Dict[str, Any]]:
        """Build query for relationship models."""
        clauses = []
        
        rel_class = self.state.model_class
        if not hasattr(rel_class, '__kuzu_rel_name__'):
            raise ValueError(
                f"Model {rel_class.__name__} is not a registered relationship - "
                f"missing __kuzu_rel_name__ attribute"
            )
        rel_name = rel_class.__kuzu_rel_name__
        
        # @@ STEP 1: Get relationship pairs from the model class
        rel_pairs = rel_class.__kuzu_relationship_pairs__
        # Apply optional subset selection for memory-safe streaming
        pairs = rel_pairs
        subset = getattr(self.state, 'pairs_subset', None)
        if subset:
            # Validate indices strictly; fail fast on invalid input
            if any((not isinstance(i, int)) or i < 0 or i >= len(rel_pairs) for i in subset):
                raise ValueError(
                    ValidationMessageConstants.INVALID_RELATIONSHIP_PAIR_INDEX
                    if hasattr(ValidationMessageConstants, 'INVALID_RELATIONSHIP_PAIR_INDEX')
                    else f"Invalid relationship pair indices: {subset}"
                )
            pairs = [rel_pairs[i] for i in subset]
        
        if not rel_pairs:
            raise ValueError(f"Relationship {rel_name} has no relationship pairs defined")
        
        # @@ STEP 2: Build MATCH patterns for all relationship pairs
        # || S.S.4: For multi-pair relationships, we need to generate multiple MATCH patterns
        rel_alias = self.state.alias
        direction = rel_class.__dict__.get('__kuzu_direction__')

        where_already_applied = False

        endpoint_aliases: List[Tuple[str, str]] = []
        # Optimization: when ALL pairs (or nearly all) are requested, skip LABEL() predicates entirely.
        # This avoids generating massive WHERE clauses with N OR conditions that cause Kuzu to hang.
        # Threshold: if requesting >= 90% of pairs, treat as "all pairs" since predicate overhead is worse.
        pairs_ratio = len(pairs) / len(rel_pairs) if rel_pairs else 1.0
        all_pairs_requested = (subset is None) or (pairs_ratio >= 0.90)
        
        if len(pairs) > 1:
            # Multi-pair relationships must enforce a *global* SKIP/LIMIT cap.
            # Using UNION ALL with per-branch SKIP/LIMIT multiplies the effective page size.
            # Instead, MATCH a single unlabeled endpoint pair and constrain endpoints by LABEL().
            from_alias = "from_node"
            to_alias = "to_node"
            endpoint_aliases.append((from_alias, to_alias))
            self.alias_map[from_alias] = from_alias
            self.alias_map[to_alias] = to_alias

            if direction:
                if direction == RelationshipDirection.FORWARD or direction == RelationshipDirection.OUTGOING:
                    pattern = f"({from_alias})-[{rel_alias}:{rel_name}]->({to_alias})"
                elif direction == RelationshipDirection.BACKWARD or direction == RelationshipDirection.INCOMING:
                    pattern = f"({from_alias})<-[{rel_alias}:{rel_name}]-({to_alias})"
                elif direction == RelationshipDirection.BOTH:
                    pattern = f"({from_alias})-[{rel_alias}:{rel_name}]-({to_alias})"
                else:
                    pattern = f"({from_alias})-[{rel_alias}:{rel_name}]->({to_alias})"
            else:
                pattern = f"({from_alias})-[{rel_alias}:{rel_name}]->({to_alias})"
            clauses.append(f"MATCH {pattern}")

            self.alias_map[rel_alias] = rel_alias
            self.alias_map[self.state.model_class.__name__] = rel_alias

            where_clause = self._build_where_clause(relationship_alias=rel_alias)
            # Only add LABEL() predicates when a subset is explicitly requested
            if all_pairs_requested:
                # No need for pair filtering - we want all relationships
                if where_clause:
                    clauses.append(where_clause)
                    where_already_applied = True
            else:
                # Decide positive vs negative predicate strategy:
                # Use negative (NOT) when requested >= half of total to minimize OR conditions
                use_negative = len(pairs) >= len(rel_pairs) // 2
                pair_clause = self._build_relationship_pair_label_predicate(
                    from_alias, to_alias, pairs, rel_pairs, use_negative=use_negative
                )
                if pair_clause:  # Could be empty if negative mode and all pairs requested
                    if where_clause:
                        base = where_clause[len("WHERE "):]
                        clauses.append(f"WHERE ({base}) AND ({pair_clause})")
                    else:
                        clauses.append(f"WHERE {pair_clause}")
                    where_already_applied = True
                elif where_clause:
                    clauses.append(where_clause)
                    where_already_applied = True
        else:
            match_patterns = []
            for pair in pairs:
                from_name = pair.get_from_name()
                to_name = pair.get_to_name()

                from_alias = "from_node"
                to_alias = "to_node"
                endpoint_aliases.append((from_alias, to_alias))

                self.alias_map[from_alias] = from_alias
                self.alias_map[to_alias] = to_alias

                if direction:
                    if direction == RelationshipDirection.FORWARD or direction == RelationshipDirection.OUTGOING:
                        pattern = f"({from_alias}:{from_name})-[{rel_alias}:{rel_name}]->({to_alias}:{to_name})"
                    elif direction == RelationshipDirection.BACKWARD or direction == RelationshipDirection.INCOMING:
                        pattern = f"({from_alias}:{from_name})<-[{rel_alias}:{rel_name}]-({to_alias}:{to_name})"
                    elif direction == RelationshipDirection.BOTH:
                        pattern = f"({from_alias}:{from_name})-[{rel_alias}:{rel_name}]-({to_alias}:{to_name})"
                    else:
                        pattern = f"({from_alias}:{from_name})-[{rel_alias}:{rel_name}]->({to_alias}:{to_name})"
                else:
                    pattern = f"({from_alias}:{from_name})-[{rel_alias}:{rel_name}]->({to_alias}:{to_name})"

                match_patterns.append(pattern)

            clauses.append(f"MATCH {match_patterns[0]}")

        self.alias_map[rel_alias] = rel_alias
        # Map relationship class name to alias for QueryField/filters referencing the class name
        rel_cls_name = self.state.model_class.__name__
        self.alias_map[rel_cls_name] = rel_alias

        if not where_already_applied:
            where_clause = self._build_where_clause(relationship_alias=rel_alias)
            if where_clause:
                clauses.append(where_clause)
        
        # @@ STEP: Check for aggregations in relationship queries
        if self.state.aggregations or self.state.group_by:
            return_items = self._build_aggregation_return()
            if self.state.group_by:
                group_items = []
                for fld in self.state.group_by:
                    if "." in fld:
                        group_items.append(fld)
                    else:
                        group_items.append(f"{self.state.alias}.{fld}")
                return_items = group_items + return_items
                # @@ STEP: Implement proper GROUP BY behavior even though Kuzu uses implicit grouping
                # || S.S: Validate that all non-aggregated fields are explicitly grouped
                # || S.S: This ensures proper SQL semantics and user control over grouping
                self._validate_group_by_semantics(return_items)
            # @@ STEP: Implement HAVING using WITH clause + WHERE pattern for relationships
            if self.state.having:
                # Build WITH items with proper aliases (Kuzu requires all expressions to be aliased)
                with_items = self._build_aggregation_return()
                if self.state.group_by:
                    group_items = []
                    for fld in self.state.group_by:
                        if "." in fld:
                            # Already has alias prefix, add AS alias
                            field_name = fld.split(".", 1)[1]
                            group_items.append(f"{fld} AS {field_name}")
                        else:
                            # Add alias prefix and AS alias
                            group_items.append(f"{self.state.alias}.{fld} AS {fld}")
                    with_items = group_items + with_items

                clauses.append(f"WITH {', '.join(with_items)}")

                # Add HAVING condition as WHERE clause (post-WITH context)
                having_cypher = self.state.having.to_cypher(self.alias_map, self.state.parameter_prefix, post_with=True)
                having_params = self.state.having.get_parameters()
                for key, value in having_params.items():
                    param_key = f"{self.state.parameter_prefix}{key}"
                    self.parameters[param_key] = value
                clauses.append(f"WHERE {having_cypher}")

                # Final RETURN with same items (aliases are now available)
                final_return_items = []
                if self.state.group_by:
                    for fld in self.state.group_by:
                        # Use the field name as alias (Kuzu requires aliases in WITH)
                        final_return_items.append(fld)

                for alias, (func, fld) in self.state.aggregations.items():
                    _ = func, fld  # Mark as intentionally unused - only alias is needed in RETURN
                    final_return_items.append(alias)

                return_clause = f"RETURN {', '.join(final_return_items)}"
            else:
                return_clause = f"RETURN {', '.join(return_items)}"
        else:
            # Build explicit RETURN including endpoints for relationship queries
            if self.state.select_fields:
                items = []
                for fld in self.state.select_fields:
                    if "." in fld:
                        items.append(fld)
                    else:
                        items.append(f"{self.state.alias}.{fld}")
                from_alias, to_alias = endpoint_aliases[0]
                return_clause = f"RETURN {', '.join(items)}, {from_alias} AS {DDLConstants.REL_FROM_NODE_FIELD}, {to_alias} AS {DDLConstants.REL_TO_NODE_FIELD}"
            else:
                from_alias, to_alias = endpoint_aliases[0]
                return_clause = f"RETURN {self.state.alias}, {from_alias} AS {DDLConstants.REL_FROM_NODE_FIELD}, {to_alias} AS {DDLConstants.REL_TO_NODE_FIELD}"
        clauses.append(return_clause)
        
        if self.state.order_by:
            order_items = []
            for fld, direction in self.state.order_by:
                if "." in fld:
                    order_items.append(f"{fld} {direction.value}")
                else:
                    order_items.append(f"{rel_alias}.{fld} {direction.value}")
            clauses.append(f"{CypherConstants.ORDER_BY} {', '.join(order_items)}")

        # @@ STEP: Kuzu requires SKIP instead of OFFSET, and SKIP must come before LIMIT
        # || S.1: Add SKIP and LIMIT as separate clauses in correct order for Kuzu
        if self.state.offset_value is not None:
            clauses.append(f"{CypherConstants.SKIP} {self.state.offset_value}")
        if self.state.limit_value is not None:
            clauses.append(f"{CypherConstants.LIMIT} {self.state.limit_value}")
        
        query = "\n".join(clauses)
        # @@ STEP: Prune parameters that are not referenced in the Cypher text
        if self.parameters:
            import re as _re
            used = set(_re.findall(r"\$([A-Za-z_][A-Za-z0-9_]*)", query))
            if used:
                self.parameters = {k: v for k, v in self.parameters.items() if k in used}
        return query, self.parameters

    def _build_relationship_pair_label_predicate(
        self,
        from_var: str,
        to_var: str,
        requested_pairs: List[Any],
        all_pairs: List[Any],
        use_negative: bool = False,
    ) -> str:
        """Build a LABEL() predicate for multi-pair relationships.
        
        When use_negative=False (positive mode):
            Build (LABEL(from)=X AND LABEL(to)=Y) OR ... for requested pairs.
        When use_negative=True (negative mode):
            Build NOT((LABEL(from)=X AND LABEL(to)=Y) OR ...) for excluded pairs.
            This is more efficient when requested_pairs > half of all_pairs.
        """
        if use_negative:
            # Build exclusion predicate for pairs NOT in requested_pairs
            requested_set = set((p.get_from_name(), p.get_to_name()) for p in requested_pairs)
            excluded_pairs = [p for p in all_pairs if (p.get_from_name(), p.get_to_name()) not in requested_set]
            if not excluded_pairs:
                # All pairs requested, no filtering needed
                return ""
            parts: List[str] = []
            for i, pair in enumerate(excluded_pairs):
                from_label = pair.get_from_name()
                to_label = pair.get_to_name()
                p_from = f"{self.state.parameter_prefix}__excl_from_{i}"
                p_to = f"{self.state.parameter_prefix}__excl_to_{i}"
                self.parameters[p_from] = from_label
                self.parameters[p_to] = to_label
                parts.append(f"(label({from_var}) = ${p_from} AND label({to_var}) = ${p_to})")
            return "NOT(" + " OR ".join(parts) + ")"
        else:
            # Positive mode: include only requested pairs
            parts: List[str] = []
            for i, pair in enumerate(requested_pairs):
                from_label = pair.get_from_name()
                to_label = pair.get_to_name()
                p_from = f"{self.state.parameter_prefix}__pairs_from_{i}"
                p_to = f"{self.state.parameter_prefix}__pairs_to_{i}"
                self.parameters[p_from] = from_label
                self.parameters[p_to] = to_label
                parts.append(f"(label({from_var}) = ${p_from} AND label({to_var}) = ${p_to})")
            return "(" + " OR ".join(parts) + ")"
    
    def _build_where_clause(self, relationship_alias: Optional[str] = None) -> str:
        """Build WHERE clause from filters."""
        if not self.state.filters:
            return ""

        conditions = []
        for filter_expr in self.state.filters:
            cypher = filter_expr.to_cypher(self.alias_map, self.state.parameter_prefix, relationship_alias)
            conditions.append(cypher)
            params = filter_expr.get_parameters()
            for key, value in params.items():
                param_key = f"{self.state.parameter_prefix}{key}"
                self.parameters[param_key] = value

        if conditions:
            return f"WHERE {' AND '.join(conditions)}"
        return ""
    
    def _build_return_clause(self) -> str:
        """Build RETURN clause."""
        if self.state.return_raw:
            # Return only known aliases instead of RETURN * to avoid exploding result width
            # Aliases registered: base node alias, join target aliases, and relationship aliases
            alias_items = []
            # Ensure deterministic order: source alias first
            if self.state.alias in self.alias_map:
                alias_items.append(self.state.alias)
            for _, v in self.alias_map.items():
                if v not in alias_items:
                    alias_items.append(v)
            return f"RETURN {', '.join(alias_items)}"

        if self.state.select_fields:
            items = []
            for fld in self.state.select_fields:
                if "." in fld:
                    items.append(fld)
                else:
                    items.append(f"{self.state.alias}.{fld}")
            return f"RETURN {('DISTINCT ' if self.state.distinct else '')}{', '.join(items)}"
        else:
            # @@ STEP: For traversals, return the target node instead of source
            return_alias = self._get_return_alias()
            return f"RETURN {('DISTINCT ' if self.state.distinct else '')}{return_alias}"

    def _get_return_alias(self) -> str:
        """Get the correct alias to return based on traversals."""
        # If return_alias is explicitly set (from traversal), use it
        if self.state.return_alias:
            return self.state.return_alias

        # Default to the original alias
        return self.state.alias
    
    def _build_aggregation_return(self) -> List[str]:
        """Build aggregation return items."""
        items = []
        for alias, (func, fld) in self.state.aggregations.items():
            # @@ STEP: Add alias prefix to field names for aggregations
            if fld != "*":
                field_with_alias = f"{self.state.alias}.{fld}"
            else:
                field_with_alias = fld
            
            if func == AggregateFunction.COUNT_DISTINCT:
                items.append(f"COUNT(DISTINCT {field_with_alias}) AS {alias}")
            elif func == AggregateFunction.COUNT:
                items.append(f"COUNT({field_with_alias}) AS {alias}")
            else:
                items.append(f"{func.value}({field_with_alias}) AS {alias}")
        return items

    def _validate_group_by_semantics(self, return_items: List[str]) -> None:
        """Validate proper GROUP BY semantics even though Kuzu uses implicit grouping.

        This ensures that:
        1. All non-aggregated fields in RETURN are explicitly in GROUP BY
        2. Users have explicit control over grouping behavior
        3. Proper SQL semantics are enforced
        4. Pure aggregations (no non-aggregated fields) are allowed without GROUP BY
        """
        if not self.state.aggregations:
            # No aggregations, no grouping validation needed
            return



        # Extract non-aggregated fields from return items
        non_aggregated_fields = []
        for item in return_items:
            # Skip aggregation functions (they contain parentheses and AS)
            if '(' not in item and ' AS ' not in item:
                # This is a regular field, not an aggregation
                field_name = item
                if '.' in field_name:
                    # Remove alias prefix (e.g., "n.department" -> "department")
                    field_name = field_name.split('.')[-1]
                non_aggregated_fields.append(field_name)

        # If there are no non-aggregated fields, this is a pure aggregation query
        # (e.g., SELECT COUNT(*) FROM table) - no GROUP BY required
        if not non_aggregated_fields:
            return

        # If there are non-aggregated fields, GROUP BY is required
        if not self.state.group_by:
            raise ValueError(
                f"GROUP BY is required when mixing aggregated and non-aggregated fields. "
                f"Non-aggregated fields found: {non_aggregated_fields}. "
                f"Either add GROUP BY for these fields or remove them from the query."
            )

        # Validate that all non-aggregated fields are in GROUP BY
        missing_fields = []
        for fld in non_aggregated_fields:
            if fld not in self.state.group_by:
                missing_fields.append(fld)

        if missing_fields:
            raise ValueError(
                f"All non-aggregated fields must be in GROUP BY clause. "
                f"Missing fields: {missing_fields}. "
                f"Current GROUP BY: {self.state.group_by}"
            )

    def _build_multi_pair_union_query(self, match_patterns: List[str], endpoint_aliases: List[Tuple[str, str]], rel_alias: str, rel_pairs: List[Any]) -> str:
        """
        Build a UNION ALL query for multi-pair relationships.

        This method creates a Cypher query that uses UNION ALL to handle
        relationships that can exist between multiple node type pairs. Each pattern
        is executed as a separate subquery, and results are combined using UNION ALL.

        The WHERE clause is applied within each subquery to filter correctly before
        the UNION. The RETURN clause explicitly returns the relationship variable and
        endpoints to maintain consistent typing across the union.

        Args:
            match_patterns: List of MATCH patterns for each relationship pair
            endpoint_aliases: List of (from_alias, to_alias) tuples for each pair
            rel_alias: Alias for the relationship in the query
            rel_pairs: List of relationship pairs (reserved for future use)

        Returns:
            Cypher query string with UNION ALL structure
        """
        # rel_pairs reserved for future use in complex relationship handling
        _ = rel_pairs

        # Build subqueries with WHERE applied within each branch for correct filtering
        # Kuzu doesn't support complex outer query wrapping for UNION, so filters
        # and ordering must be applied within each subquery
        final_subqueries = []

        for (pattern, (from_alias, to_alias)) in zip(match_patterns, endpoint_aliases):
            # || S.S.6: Rebuild each subquery with proper filtering and ordering
            subquery_clauses = [f"MATCH {pattern}"]

            # || S.S.7: Add WHERE clause if filters exist
            where_clause = self._build_where_clause(relationship_alias=rel_alias)
            if where_clause:
                subquery_clauses.append(where_clause)

            # || S.S.8: Return the full relationship variable for consistent typing across UNIONs
            # || Returning the variable ensures stable column types; include endpoints explicitly
            # Use neutral endpoint aliases to avoid binder conflicts in UNION
            # Normalize endpoint typing across UNION branches by rematching endpoints via ID to unlabeled variables
            subquery_clauses.append(f"WITH {from_alias}, {to_alias}, {rel_alias}")
            subquery_clauses.append(f"MATCH (f) WHERE ID(f) = ID({from_alias})")
            subquery_clauses.append(f"MATCH (t) WHERE ID(t) = ID({to_alias})")
            return_line = (
                f"RETURN {rel_alias} AS {self.state.alias}, "
                f"f AS {QueryReturnAliasConstants.FROM_ENDPOINT}, "
                f"t AS {QueryReturnAliasConstants.TO_ENDPOINT}, "
                f"ID(f) AS {QueryReturnAliasConstants.FROM_ID}, "
                f"ID(t) AS {QueryReturnAliasConstants.TO_ID}"
            )
            subquery_clauses.append(return_line)

            # || S.S.9: Add ORDER BY if specified
            if self.state.order_by:
                order_items = []
                for ob_field, direction in self.state.order_by:
                    if "." in ob_field:
                        order_items.append(f"{ob_field} {direction.value}")
                    else:
                        order_items.append(f"{rel_alias}.{ob_field} {direction.value}")
                subquery_clauses.append(f"{CypherConstants.ORDER_BY} {', '.join(order_items)}")

            # || S.S.10: Add SKIP and LIMIT if specified
            if self.state.offset_value is not None:
                subquery_clauses.append(f"{CypherConstants.SKIP} {self.state.offset_value}")
            if self.state.limit_value is not None:
                subquery_clauses.append(f"{CypherConstants.LIMIT} {self.state.limit_value}")

            final_subqueries.append(" ".join(subquery_clauses))

        # || S.S.11: Clear problematic parameters that don't exist in the query
        # || The node aliases are hardcoded in the patterns, not parameterized
        params_to_remove = []
        for param_name in self.parameters:
            if param_name.startswith(('from_node_', 'to_node_')):
                params_to_remove.append(param_name)

        for param_name in params_to_remove:
            del self.parameters[param_name]

        # || S.S.12: Combine with UNION ALL
        return f" {CypherConstants.UNION_ALL} ".join(final_subqueries)

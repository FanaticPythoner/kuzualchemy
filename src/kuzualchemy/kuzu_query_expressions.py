# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
Filter expressions and operators for KuzuAlchemy ORM.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .constants import CypherConstants


class ComparisonOperator(Enum):
    """Supported comparison operators for queries."""
    EQ = CypherConstants.EQ
    NEQ = CypherConstants.NEQ
    LT = CypherConstants.LT
    LTE = CypherConstants.LTE
    GT = CypherConstants.GT
    GTE = CypherConstants.GTE
    IN = CypherConstants.IN
    NOT_IN = "NOT IN"  # This specific combination not in CypherConstants
    LIKE = "=~"  # Kuzu-specific regex operator
    NOT_LIKE = "NOT_LIKE"  # Kuzu-specific regex operator (handled specially)
    IS_NULL = CypherConstants.IS_NULL
    IS_NOT_NULL = CypherConstants.IS_NOT_NULL
    CONTAINS = CypherConstants.CONTAINS
    STARTS_WITH = CypherConstants.STARTS_WITH
    ENDS_WITH = CypherConstants.ENDS_WITH
    EXISTS = CypherConstants.EXISTS
    NOT_EXISTS = "NOT EXISTS"  # This specific combination not in CypherConstants
    REGEX = "=~"  # Kuzu-specific regex operator
    NOT_REGEX = "NOT_REGEX"  # Kuzu-specific regex operator (handled specially)
    REGEX_MATCH = "=~"  # Kuzu regex match operator
    NOT_REGEX_MATCH = "!~"  # Kuzu negative regex match operator


class LogicalOperator(Enum):
    """Logical operators for combining conditions."""
    AND = CypherConstants.AND
    OR = CypherConstants.OR
    NOT = CypherConstants.NOT
    XOR = CypherConstants.XOR


class AggregateFunction(Enum):
    """Aggregate functions for queries."""
    COUNT = CypherConstants.COUNT
    COUNT_DISTINCT = "COUNT(DISTINCT {})"  # Template format, not a direct constant
    SUM = "SUM"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"
    COLLECT = "COLLECT"
    COLLECT_SET = "COLLECT_SET"
    GROUP_CONCAT = "STRING_AGG"
    STDDEV = "STDDEV"
    VARIANCE = "VAR"
    PERCENTILE = "PERCENTILE_CONT"
    MEDIAN = "MEDIAN"
    MODE = "MODE"


class OrderDirection(Enum):
    """Order directions for sorting."""
    ASC = "ASC"
    DESC = "DESC"
    NULLS_FIRST = "NULLS FIRST"
    NULLS_LAST = "NULLS LAST"


class JoinType(Enum):
    """Join types for relationship traversal."""
    INNER = ""
    OPTIONAL = "OPTIONAL"
    MANDATORY = "WHERE EXISTS"


class ArithmeticOperator(Enum):
    """Arithmetic operators for numeric expressions."""
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"
    POW = "^"


class StringOperator(Enum):
    """String operators for text expressions."""
    CONCAT = "+"
    INDEX = "[]"
    SLICE = "[:]"


class PatternOperator(Enum):
    """Pattern matching operators."""
    REGEX_MATCH = "=~"
    NOT_REGEX_MATCH = "!~"


class TemporalOperator(Enum):
    """Temporal operators for date/time/interval arithmetic."""
    DATE_ADD_INT = "DATE_ADD_INT"  # DATE + INT64
    DATE_ADD_INTERVAL = "DATE_ADD_INTERVAL"  # DATE + INTERVAL
    DATE_SUB_DATE = "DATE_SUB_DATE"  # DATE - DATE
    DATE_SUB_INTERVAL = "DATE_SUB_INTERVAL"  # DATE - INTERVAL

    TIMESTAMP_ADD_INTERVAL = "TIMESTAMP_ADD_INTERVAL"  # TIMESTAMP + INTERVAL
    TIMESTAMP_SUB_TIMESTAMP = "TIMESTAMP_SUB_TIMESTAMP"  # TIMESTAMP - TIMESTAMP
    TIMESTAMP_SUB_INTERVAL = "TIMESTAMP_SUB_INTERVAL"  # TIMESTAMP - INTERVAL

    INTERVAL_ADD_INTERVAL = "INTERVAL_ADD_INTERVAL"  # INTERVAL + INTERVAL
    INTERVAL_ADD_DATE = "INTERVAL_ADD_DATE"  # INTERVAL + DATE
    INTERVAL_ADD_TIMESTAMP = "INTERVAL_ADD_TIMESTAMP"  # INTERVAL + TIMESTAMP
    INTERVAL_SUB_INTERVAL = "INTERVAL_SUB_INTERVAL"  # INTERVAL - INTERVAL
    INTERVAL_SUB_DATE = "INTERVAL_SUB_DATE"  # DATE - INTERVAL (reverse)
    INTERVAL_SUB_TIMESTAMP = "INTERVAL_SUB_TIMESTAMP"  # TIMESTAMP - INTERVAL (reverse)


class FilterExpression(ABC):
    """Abstract base class for filter expressions."""
    
    @abstractmethod
    def to_cypher(self, alias_map: Dict[str, str], param_prefix: str = "", relationship_alias: Optional[str] = None, post_with: bool = False) -> str:
        """Convert expression to Cypher WHERE clause fragment.

        Args:
            alias_map: Mapping of aliases
            param_prefix: Prefix for parameters
            relationship_alias: Alias for relationship queries
            post_with: True if this is being used after a WITH clause (for HAVING)
        """
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get query parameters for this expression."""
    
    @abstractmethod
    def get_field_references(self) -> Set[str]:
        """Get all field references in this expression."""
    
    def __and__(self, other: "FilterExpression") -> "FilterExpression":
        """Combine with AND operator."""
        return CompoundFilterExpression(LogicalOperator.AND, [self, other])
    
    def __or__(self, other: "FilterExpression") -> "FilterExpression":
        """Combine with OR operator."""
        return CompoundFilterExpression(LogicalOperator.OR, [self, other])
    
    def __invert__(self) -> "FilterExpression":
        """Apply NOT operator."""
        return NotFilterExpression(self)
    
    def __xor__(self, other: "FilterExpression") -> "FilterExpression":
        """Combine with XOR operator."""
        return CompoundFilterExpression(LogicalOperator.XOR, [self, other])


class FieldFilterExpression(FilterExpression):
    """Filter expression for a single field comparison."""
    
    def __init__(
        self,
        field_path: str,
        operator: ComparisonOperator,
        value: Any = None,
        parameter_name: Optional[str] = None,
        case_sensitive: bool = True
    ):
        self.field_path = field_path
        self.operator = operator
        self.value = value
        self.parameter_name = parameter_name or f"param_{abs(hash((field_path, operator, str(value))))}"
        self.case_sensitive = case_sensitive
    
    def to_cypher(self, alias_map: Dict[str, str], param_prefix: str = "", relationship_alias: Optional[str] = None, post_with: bool = False) -> str:
        """Convert to Cypher WHERE clause fragment."""
        # Resolve left-hand side field reference
        parts = self.field_path.split(".", 1)
        if len(parts) == 2:
            alias, field = parts
            if alias in alias_map:
                mapped_alias = alias_map[alias]
            else:
                mapped_alias = next(iter(alias_map.values())) if alias_map else alias
            field_ref = f"{mapped_alias}.{field}"
        else:
            # @@ STEP: Handle post-WITH context (HAVING clauses)
            if post_with:
                # In post-WITH context, use bare field names (they are aliases from WITH clause)
                field_ref = self.field_path
            elif relationship_alias:
                # @@ STEP: For relationship queries, use relationship alias for unqualified fields
                # || S.S: This fixes the core issue where relationship properties were being
                # || looked up on nodes instead of on the relationship itself
                field_ref = f"{relationship_alias}.{self.field_path}"
            else:
                default_alias = next(iter(alias_map.values())) if alias_map else "n"
                field_ref = f"{default_alias}.{self.field_path}"

        # Field-to-field comparisons: if RHS is a QueryField, inline it (no parameter)
        from .kuzu_query_fields import QueryField as _QF
        if isinstance(self.value, _QF):
            rhs_parts = self.value.field_path.split(".", 1)
            if len(rhs_parts) == 2:
                r_alias, r_field = rhs_parts
                if r_alias in alias_map:
                    r_mapped = alias_map[r_alias]
                else:
                    r_mapped = next(iter(alias_map.values())) if alias_map else r_alias
                rhs_ref = f"{r_mapped}.{r_field}"
            else:
                if relationship_alias:
                    rhs_ref = f"{relationship_alias}.{self.value.field_path}"
                elif post_with:
                    rhs_ref = self.value.field_path
                else:
                    default_alias = next(iter(alias_map.values())) if alias_map else "n"
                    rhs_ref = f"{default_alias}.{self.value.field_path}"

            if not self.case_sensitive and self.operator in (
                ComparisonOperator.EQ, ComparisonOperator.NEQ,
                ComparisonOperator.LIKE, ComparisonOperator.NOT_LIKE,
                ComparisonOperator.STARTS_WITH, ComparisonOperator.ENDS_WITH
            ):
                return f"LOWER({field_ref}) {self.operator.value} LOWER({rhs_ref})"
            return f"{field_ref} {self.operator.value} {rhs_ref}"

        param_name = f"{param_prefix}{self.parameter_name}"

        if not self.case_sensitive and self.operator in (
            ComparisonOperator.EQ, ComparisonOperator.NEQ,
            ComparisonOperator.LIKE, ComparisonOperator.NOT_LIKE,
            ComparisonOperator.STARTS_WITH, ComparisonOperator.ENDS_WITH
        ):
            field_ref = f"LOWER({field_ref})"
            param_name = f"LOWER(${param_name})"

        if self.operator == ComparisonOperator.IS_NULL:
            return f"{field_ref} IS NULL"
        elif self.operator == ComparisonOperator.IS_NOT_NULL:
            return f"{field_ref} IS NOT NULL"
        elif self.operator == ComparisonOperator.IN:
            return f"{field_ref} IN ${param_name}"
        elif self.operator == ComparisonOperator.NOT_IN:
            return f"NOT {field_ref} IN ${param_name}"
        elif self.operator == ComparisonOperator.CONTAINS:
            # @@ STEP: Use regex pattern matching for CONTAINS
            # || S.S: We need to embed the value directly in the regex pattern
            value = self.value
            if isinstance(value, str):
                # || S.S: Escape special regex characters in the value
                escaped_value = re.escape(value)
                return f"{field_ref} =~ '.*{escaped_value}.*'"
            elif isinstance(value, (int, float, bool)):
                # Handle numeric and boolean values by converting to string and escaping
                escaped_value = re.escape(str(value))
                return f"{field_ref} =~ '.*{escaped_value}.*'"
            else:
                # Handle all other types as parameters
                return f"{field_ref} =~ ${param_name}"
        elif self.operator == ComparisonOperator.STARTS_WITH:
            return f"{field_ref} STARTS WITH ${param_name}"
        elif self.operator == ComparisonOperator.ENDS_WITH:
            return f"{field_ref} ENDS WITH ${param_name}"
        elif self.operator in (ComparisonOperator.LIKE, ComparisonOperator.REGEX, ComparisonOperator.REGEX_MATCH):
            return f"{field_ref} =~ ${param_name}"
        elif self.operator in (ComparisonOperator.NOT_LIKE, ComparisonOperator.NOT_REGEX):
            return f"NOT {field_ref} =~ ${param_name}"
        elif self.operator == ComparisonOperator.NOT_REGEX_MATCH:
            return f"{field_ref} !~ ${param_name}"
        elif self.operator == ComparisonOperator.EXISTS:
            return f"EXISTS({field_ref})"
        elif self.operator == ComparisonOperator.NOT_EXISTS:
            return f"NOT EXISTS({field_ref})"
        else:
            return f"{field_ref} {self.operator.value} ${param_name}"

    def get_parameters(self) -> Dict[str, Any]:
        """Get query parameters."""
        if self.operator in (
            ComparisonOperator.IS_NULL, ComparisonOperator.IS_NOT_NULL,
            ComparisonOperator.EXISTS, ComparisonOperator.NOT_EXISTS
        ):
            return {}

        # @@ STEP: For CONTAINS with string values, no parameter is used (value is embedded)
        if self.operator == ComparisonOperator.CONTAINS and isinstance(self.value, str):
            return {}

        # @@ STEP: Do not bind QueryField as a parameter; it is inlined in to_cypher
        from .kuzu_query_fields import QueryField as _QF
        if isinstance(self.value, _QF):
            return {}

        # @@ STEP: If RHS is an expression, collect its parameters only
        if isinstance(self.value, (ArithmeticExpression, TemporalExpression, FunctionExpression, PatternExpression)):
            return self.value.get_parameters()

        value = self.value
        if not self.case_sensitive and type(value) is str:
            value = value.lower()

        return {self.parameter_name: value}

    def get_field_references(self) -> Set[str]:
        """Get field references."""
        return {self.field_path}


class CompoundFilterExpression(FilterExpression):
    """Compound filter expression combining multiple expressions."""
    
    def __init__(self, operator: LogicalOperator, expressions: List[FilterExpression]):
        self.operator = operator
        self.expressions = expressions
    
    def to_cypher(self, alias_map: Dict[str, str], param_prefix: str = "", relationship_alias: Optional[str] = None, post_with: bool = False) -> str:
        """Convert to Cypher WHERE clause fragment."""
        if not self.expressions:
            return "TRUE"

        sub_expressions = [
            expr.to_cypher(alias_map, param_prefix, relationship_alias, post_with)
            for expr in self.expressions
        ]
        
        if self.operator == LogicalOperator.NOT:
            return f"NOT ({sub_expressions[0]})"
        elif self.operator == LogicalOperator.XOR:
            if len(sub_expressions) != 2:
                raise ValueError("XOR requires exactly 2 expressions")
            return f"(({sub_expressions[0]}) AND NOT ({sub_expressions[1]})) OR (NOT ({sub_expressions[0]}) AND ({sub_expressions[1]}))"
        else:
            return f" {self.operator.value} ".join(f"({expr})" for expr in sub_expressions)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get combined parameters from all expressions."""
        params = {}
        for expr in self.expressions:
            params.update(expr.get_parameters())
        return params
    
    def get_field_references(self) -> Set[str]:
        """Get all field references."""
        refs = set()
        for expr in self.expressions:
            refs.update(expr.get_field_references())
        return refs


class NotFilterExpression(FilterExpression):
    """NOT filter expression."""
    
    def __init__(self, expression: FilterExpression):
        self.expression = expression
    
    def to_cypher(self, alias_map: Dict[str, str], param_prefix: str = "", relationship_alias: Optional[str] = None, post_with: bool = False) -> str:
        """Convert to Cypher WHERE clause fragment."""
        return f"NOT ({self.expression.to_cypher(alias_map, param_prefix, relationship_alias, post_with)})"
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get parameters from wrapped expression."""
        return self.expression.get_parameters()
    
    def get_field_references(self) -> Set[str]:
        """Get field references."""
        return self.expression.get_field_references()


class RawCypherExpression(FilterExpression):
    """Raw Cypher expression for advanced queries."""
    
    def __init__(self, cypher: str, parameters: Optional[Dict[str, Any]] = None):
        self.cypher = cypher
        self.parameters = parameters or {}
    
    def to_cypher(self, alias_map: Dict[str, str], param_prefix: str = "", relationship_alias: Optional[str] = None, post_with: bool = False) -> str:
        """Return raw Cypher with alias substitution."""
        result = self.cypher
        for alias, mapped in alias_map.items():
            result = re.sub(rf'\{{\{{{alias}\}}\}}', mapped, result)
        for param in self.parameters:
            result = result.replace(f"${param}", f"${param_prefix}{param}")
        return result
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get parameters."""
        return self.parameters
    
    def get_field_references(self) -> Set[str]:
        """Cannot determine field references from raw Cypher."""
        return set()


class BetweenExpression(FilterExpression):
    """BETWEEN filter expression."""
    
    def __init__(self, field_path: str, lower: Any, upper: Any, inclusive: bool = True):
        self.field_path = field_path
        self.lower = lower
        self.upper = upper
        self.inclusive = inclusive
        self.lower_param = f"{field_path.replace('.', '_')}_lower"
        self.upper_param = f"{field_path.replace('.', '_')}_upper"
    
    def to_cypher(self, alias_map: Dict[str, str], param_prefix: str = "", relationship_alias: Optional[str] = None, post_with: bool = False) -> str:
        """Convert to Cypher WHERE clause fragment."""
        parts = self.field_path.split(".", 1)
        if len(parts) == 2:
            alias, field = parts
            mapped_alias = alias_map[alias] if alias in alias_map else (next(iter(alias_map.values())) if alias_map else alias)
            field_ref = f"{mapped_alias}.{field}"
        else:
            # @@ STEP: Handle post-WITH context (HAVING clauses)
            if post_with:
                # In post-WITH context, use bare field names (they are aliases from WITH clause)
                field_ref = self.field_path
            elif relationship_alias:
                # @@ STEP: For relationship queries, use relationship alias for unqualified fields
                field_ref = f"{relationship_alias}.{self.field_path}"
            else:
                default_alias = next(iter(alias_map.values())) if alias_map else "n"
                field_ref = f"{default_alias}.{self.field_path}"

        lower_op = ">=" if self.inclusive else ">"
        upper_op = "<=" if self.inclusive else "<"

        return f"({field_ref} {lower_op} ${param_prefix}{self.lower_param} AND {field_ref} {upper_op} ${param_prefix}{self.upper_param})"
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get parameters."""
        return {
            self.lower_param: self.lower,
            self.upper_param: self.upper
        }
    
    def get_field_references(self) -> Set[str]:
        """Get field references."""
        return {self.field_path}


class ArithmeticExpression(FilterExpression):
    """Arithmetic expression for numeric operations."""

    _param_counter = 0

    def __init__(self, left: Any, operator: ArithmeticOperator, right: Any):
        self.left = left
        self.operator = operator
        self.right = right

        # Use deterministic parameter names without hash collisions
        ArithmeticExpression._param_counter += 1
        self.param_id = ArithmeticExpression._param_counter
        self.parameter_name_left = f"arith_left_{self.param_id}"
        self.parameter_name_right = f"arith_right_{self.param_id}"

    def to_cypher(self, alias_map: Dict[str, str], param_prefix: str = "", relationship_alias: Optional[str] = None, post_with: bool = False) -> str:
        """Convert to Cypher arithmetic expression."""
        left_expr = self._format_operand(self.left, alias_map, param_prefix, relationship_alias, post_with)
        right_expr = self._format_operand(self.right, alias_map, param_prefix, relationship_alias, post_with)

        return f"({left_expr} {self.operator.value} {right_expr})"

    def _format_operand(self, operand: Any, alias_map: Dict[str, str], param_prefix: str, relationship_alias: Optional[str], post_with: bool) -> str:
        """Format an operand for Cypher output."""
        from .kuzu_query_fields import QueryField

        if isinstance(operand, QueryField):
            # Handle QueryField operands
            parts = operand.field_path.split(".", 1)
            if len(parts) == 2:
                alias, field = parts
                mapped_alias = alias_map[alias] if alias in alias_map else (next(iter(alias_map.values())) if alias_map else alias)
                return f"{mapped_alias}.{field}"
            else:
                if post_with:
                    return operand.field_path
                elif relationship_alias:
                    return f"{relationship_alias}.{operand.field_path}"
                else:
                    default_alias = next(iter(alias_map.values())) if alias_map else "n"
                    return f"{default_alias}.{operand.field_path}"
        elif isinstance(operand, (ArithmeticExpression, FunctionExpression, TemporalExpression, PatternExpression)):
            # Handle nested expressions
            return operand.to_cypher(alias_map, param_prefix, relationship_alias, post_with)
        elif isinstance(operand, (int, float)):
            # Handle numeric literals directly
            return str(operand)
        elif isinstance(operand, str):
            # Handle string parameters
            if operand is self.left:
                param_name = self.parameter_name_left
            elif operand is self.right:
                param_name = self.parameter_name_right
            else:
                raise ValueError(f"String operand '{operand}' is not left or right operand of ArithmeticExpression")
            return f"${param_prefix}{param_name}"
        elif isinstance(operand, (list, tuple, dict)):
            # Handle collection parameters
            if operand is self.left:
                param_name = self.parameter_name_left
            elif operand is self.right:
                param_name = self.parameter_name_right
            else:
                raise ValueError(f"Collection operand '{operand}' is not left or right operand of ArithmeticExpression")
            return f"${param_prefix}{param_name}"
        else:
            # Handle any other parameter types
            if operand is self.left:
                param_name = self.parameter_name_left
            elif operand is self.right:
                param_name = self.parameter_name_right
            else:
                raise ValueError(f"Operand '{operand}' of type {type(operand)} is not left or right operand of ArithmeticExpression")
            return f"${param_prefix}{param_name}"

    def get_parameters(self) -> Dict[str, Any]:
        """Get query parameters."""
        params = {}

        # Add parameters for non-literal operands
        from .kuzu_query_fields import QueryField

        if not isinstance(self.left, (int, float)) and not isinstance(self.left, QueryField):
            params[self.parameter_name_left] = self.left
        if not isinstance(self.right, (int, float)) and not isinstance(self.right, QueryField):
            params[self.parameter_name_right] = self.right

        # Add parameters from nested expressions
        if isinstance(self.left, (ArithmeticExpression, TemporalExpression, FunctionExpression, PatternExpression)):
            params.update(self.left.get_parameters())
        if isinstance(self.right, (ArithmeticExpression, TemporalExpression, FunctionExpression, PatternExpression)):
            params.update(self.right.get_parameters())

        return params

    def get_field_references(self) -> Set[str]:
        """Get field references."""
        from .kuzu_query_fields import QueryField
        refs = set()

        if isinstance(self.left, QueryField):
            refs.add(self.left.field_path)
        elif isinstance(self.left, (ArithmeticExpression, TemporalExpression, FunctionExpression, PatternExpression)):
            refs.update(self.left.get_field_references())

        if isinstance(self.right, QueryField):
            refs.add(self.right.field_path)
        elif isinstance(self.right, (ArithmeticExpression, TemporalExpression, FunctionExpression, PatternExpression)):
            refs.update(self.right.get_field_references())

        return refs

    # ============================================================================
    # ARITHMETIC OPERATORS FOR ARITHMETIC EXPRESSIONS
    # ============================================================================

    def __add__(self, other: Any) -> 'ArithmeticExpression':
        """Addition operator (+)."""
        return ArithmeticExpression(self, ArithmeticOperator.ADD, other)

    def __radd__(self, other: Any) -> 'ArithmeticExpression':
        """Right addition operator (+)."""
        return ArithmeticExpression(other, ArithmeticOperator.ADD, self)

    def __sub__(self, other: Any) -> 'ArithmeticExpression':
        """Subtraction operator (-)."""
        return ArithmeticExpression(self, ArithmeticOperator.SUB, other)

    def __rsub__(self, other: Any) -> 'ArithmeticExpression':
        """Right subtraction operator (-)."""
        return ArithmeticExpression(other, ArithmeticOperator.SUB, self)

    # ============================================================================
    # COMPARISON OPERATORS FOR FUNCTION EXPRESSIONS
    # ============================================================================

    def _cmp(self, op: ComparisonOperator, other: Any) -> 'FunctionFilterExpression':
        return FunctionFilterExpression(self, op, other)

    def __eq__(self, other: Any) -> 'FunctionFilterExpression':  # type: ignore[override]
        return self._cmp(ComparisonOperator.EQ, other)

    def __ne__(self, other: Any) -> 'FunctionFilterExpression':  # type: ignore[override]
        return self._cmp(ComparisonOperator.NEQ, other)

    def __lt__(self, other: Any) -> 'FunctionFilterExpression':
        return self._cmp(ComparisonOperator.LT, other)

    def __le__(self, other: Any) -> 'FunctionFilterExpression':
        return self._cmp(ComparisonOperator.LTE, other)

    def __gt__(self, other: Any) -> 'FunctionFilterExpression':
        return self._cmp(ComparisonOperator.GT, other)

    def __ge__(self, other: Any) -> 'FunctionFilterExpression':
        return self._cmp(ComparisonOperator.GTE, other)

    def like(self, pattern: str) -> 'FunctionFilterExpression':
        return self._cmp(ComparisonOperator.LIKE, pattern)

    def not_like(self, pattern: str) -> 'FunctionFilterExpression':
        return self._cmp(ComparisonOperator.NOT_LIKE, pattern)

    def regex(self, pattern: str) -> 'FunctionFilterExpression':
        return self._cmp(ComparisonOperator.REGEX_MATCH, pattern)

    def not_regex(self, pattern: str) -> 'FunctionFilterExpression':
        return self._cmp(ComparisonOperator.NOT_REGEX_MATCH, pattern)

    def in_(self, values: list[Any]) -> 'FunctionFilterExpression':
        return self._cmp(ComparisonOperator.IN, values)

    def not_in(self, values: list[Any]) -> 'FunctionFilterExpression':
        return self._cmp(ComparisonOperator.NOT_IN, values)

    def is_null(self) -> 'FunctionFilterExpression':
        return self._cmp(ComparisonOperator.IS_NULL, None)

    def is_not_null(self) -> 'FunctionFilterExpression':
        return self._cmp(ComparisonOperator.IS_NOT_NULL, None)

    def __mul__(self, other: Any) -> 'ArithmeticExpression':
        """Multiplication operator (*)."""
        return ArithmeticExpression(self, ArithmeticOperator.MUL, other)

    def __rmul__(self, other: Any) -> 'ArithmeticExpression':
        """Right multiplication operator (*)."""
        return ArithmeticExpression(other, ArithmeticOperator.MUL, self)

    def __truediv__(self, other: Any) -> 'ArithmeticExpression':
        """Division operator (/)."""
        return ArithmeticExpression(self, ArithmeticOperator.DIV, other)

    def __rtruediv__(self, other: Any) -> 'ArithmeticExpression':
        """Right division operator (/)."""
        return ArithmeticExpression(other, ArithmeticOperator.DIV, self)

    def __mod__(self, other: Any) -> 'ArithmeticExpression':
        """Modulo operator (%)."""
        return ArithmeticExpression(self, ArithmeticOperator.MOD, other)

    def __rmod__(self, other: Any) -> 'ArithmeticExpression':
        """Right modulo operator (%)."""
        return ArithmeticExpression(other, ArithmeticOperator.MOD, self)

    def __pow__(self, other: Any) -> 'ArithmeticExpression':
        """Power operator (^)."""
        return ArithmeticExpression(self, ArithmeticOperator.POW, other)

    def __rpow__(self, other: Any) -> 'ArithmeticExpression':
        """Right power operator (^)."""
        return ArithmeticExpression(other, ArithmeticOperator.POW, self)

    # ============================================================================
    # INDEXING AND SLICING SUPPORT FOR ARITHMETIC EXPRESSIONS
    # ============================================================================

    def __getitem__(self, key: Any) -> 'FunctionExpression':
        """String/array indexing and slicing operator [] for arithmetic expressions."""
        if isinstance(key, slice):
            # Handle slicing [start:end]
            start = key.start if key.start is not None else 1  # Kuzu uses 1-based indexing
            stop = key.stop if key.stop is not None else -1
            return FunctionExpression("array_slice", [self, start, stop])
        else:
            # Handle indexing [index]
            return FunctionExpression("array_extract", [self, key])


class TemporalExpression(FilterExpression):
    """Temporal expression for date/time/interval operations."""

    _param_counter = 0

    def __init__(self, left: Any, operator: TemporalOperator, right: Any):
        self.left = left
        self.operator = operator
        self.right = right

        # Use deterministic parameter names without hash collisions
        TemporalExpression._param_counter += 1
        self.param_id = TemporalExpression._param_counter
        self.parameter_name_left = f"temp_left_{self.param_id}"
        self.parameter_name_right = f"temp_right_{self.param_id}"

    def to_cypher(self, alias_map: Dict[str, str], param_prefix: str = "", relationship_alias: Optional[str] = None, post_with: bool = False) -> str:
        """Convert to Cypher temporal expression."""
        left_expr = self._format_operand(self.left, alias_map, param_prefix, relationship_alias, post_with)
        right_expr = self._format_operand(self.right, alias_map, param_prefix, relationship_alias, post_with)

        # All temporal operations use + or - operators in Cypher
        if self.operator in (
            TemporalOperator.DATE_ADD_INT, TemporalOperator.DATE_ADD_INTERVAL,
            TemporalOperator.TIMESTAMP_ADD_INTERVAL, TemporalOperator.INTERVAL_ADD_INTERVAL,
            TemporalOperator.INTERVAL_ADD_DATE, TemporalOperator.INTERVAL_ADD_TIMESTAMP
        ):
            return f"({left_expr} + {right_expr})"
        else:  # All subtraction operations
            return f"({left_expr} - {right_expr})"

    def _format_operand(self, operand: Any, alias_map: Dict[str, str], param_prefix: str, relationship_alias: Optional[str], post_with: bool) -> str:
        """Format an operand for Cypher output."""
        from .kuzu_query_fields import QueryField

        if isinstance(operand, QueryField):
            # Handle QueryField operands
            parts = operand.field_path.split(".", 1)
            if len(parts) == 2:
                alias, field = parts
                mapped_alias = alias_map[alias] if alias in alias_map else (next(iter(alias_map.values())) if alias_map else alias)
                return f"{mapped_alias}.{field}"
            else:
                if post_with:
                    return operand.field_path
                elif relationship_alias:
                    return f"{relationship_alias}.{operand.field_path}"
                else:
                    default_alias = next(iter(alias_map.values())) if alias_map else "n"
                    return f"{default_alias}.{operand.field_path}"
        elif isinstance(operand, (ArithmeticExpression, TemporalExpression, FunctionExpression, PatternExpression)):
            # Handle nested expressions
            return operand.to_cypher(alias_map, param_prefix, relationship_alias, post_with)
        elif isinstance(operand, (int, float)):
            # Handle numeric literals directly
            return str(operand)
        elif isinstance(operand, str):
            # Handle string literals (for dates, timestamps, intervals)
            return f"'{operand}'"
        elif isinstance(operand, (list, tuple, dict)):
            # Handle collection parameters
            if operand is self.left:
                param_name = self.parameter_name_left
            elif operand is self.right:
                param_name = self.parameter_name_right
            else:
                raise ValueError(f"Collection operand '{operand}' is not left or right operand of TemporalExpression")
            return f"${param_prefix}{param_name}"
        else:
            # Handle any other parameter types
            if operand is self.left:
                param_name = self.parameter_name_left
            elif operand is self.right:
                param_name = self.parameter_name_right
            else:
                raise ValueError(f"Operand '{operand}' of type {type(operand)} is not left or right operand of TemporalExpression")
            return f"${param_prefix}{param_name}"

    def get_parameters(self) -> Dict[str, Any]:
        """Get query parameters."""
        params = {}

        # Add parameters for non-literal operands
        # For temporal expressions, we need to handle dynamic parameters differently
        from .kuzu_query_fields import QueryField

        if not isinstance(self.left, (int, float)) and not isinstance(self.left, QueryField):
            # Check if it's a temporal literal (DATE, TIMESTAMP, INTERVAL strings)
            if isinstance(self.left, str) and any(keyword in self.left.upper() for keyword in ['DATE(', 'TIMESTAMP(', 'INTERVAL(']):
                # It's a temporal literal, don't parameterize
                pass
            elif not isinstance(self.left, str):
                params[self.parameter_name_left] = self.left
            else:
                # It's a dynamic string parameter
                params[self.parameter_name_left] = self.left

        if not isinstance(self.right, (int, float)) and not isinstance(self.right, QueryField):
            # Check if it's a temporal literal (DATE, TIMESTAMP, INTERVAL strings)
            if isinstance(self.right, str) and any(keyword in self.right.upper() for keyword in ['DATE(', 'TIMESTAMP(', 'INTERVAL(']):
                # It's a temporal literal, don't parameterize
                pass
            elif not isinstance(self.right, str):
                params[self.parameter_name_right] = self.right
            else:
                # It's a dynamic string parameter
                params[self.parameter_name_right] = self.right

        # Add parameters from nested expressions
        if isinstance(self.left, (ArithmeticExpression, TemporalExpression, FunctionExpression, PatternExpression)):
            params.update(self.left.get_parameters())
        if isinstance(self.right, (ArithmeticExpression, TemporalExpression, FunctionExpression, PatternExpression)):
            params.update(self.right.get_parameters())

        return params

    def get_field_references(self) -> Set[str]:
        """Get field references."""
        from .kuzu_query_fields import QueryField
        refs = set()

        if isinstance(self.left, QueryField):
            refs.add(self.left.field_path)
        elif isinstance(self.left, (ArithmeticExpression, TemporalExpression, FunctionExpression, PatternExpression)):
            refs.update(self.left.get_field_references())

        if isinstance(self.right, QueryField):
            refs.add(self.right.field_path)
        elif isinstance(self.right, (ArithmeticExpression, TemporalExpression, FunctionExpression, PatternExpression)):
            refs.update(self.right.get_field_references())

        return refs

    # ============================================================================
    # ARITHMETIC OPERATORS FOR TEMPORAL EXPRESSIONS
    # ============================================================================

    def __add__(self, other: Any) -> 'ArithmeticExpression':
        """Addition operator (+)."""
        return ArithmeticExpression(self, ArithmeticOperator.ADD, other)

    def __radd__(self, other: Any) -> 'ArithmeticExpression':
        """Right addition operator (+)."""
        return ArithmeticExpression(other, ArithmeticOperator.ADD, self)

    def __sub__(self, other: Any) -> 'ArithmeticExpression':
        """Subtraction operator (-)."""
        return ArithmeticExpression(self, ArithmeticOperator.SUB, other)

    def __rsub__(self, other: Any) -> 'ArithmeticExpression':
        """Right subtraction operator (-)."""
        return ArithmeticExpression(other, ArithmeticOperator.SUB, self)

    def __mul__(self, other: Any) -> 'ArithmeticExpression':
        """Multiplication operator (*)."""
        return ArithmeticExpression(self, ArithmeticOperator.MUL, other)

    def __rmul__(self, other: Any) -> 'ArithmeticExpression':
        """Right multiplication operator (*)."""
        return ArithmeticExpression(other, ArithmeticOperator.MUL, self)

    def __truediv__(self, other: Any) -> 'ArithmeticExpression':
        """Division operator (/)."""
        return ArithmeticExpression(self, ArithmeticOperator.DIV, other)

    def __rtruediv__(self, other: Any) -> 'ArithmeticExpression':
        """Right division operator (/)."""
        return ArithmeticExpression(other, ArithmeticOperator.DIV, self)

    def __mod__(self, other: Any) -> 'ArithmeticExpression':
        """Modulo operator (%)."""
        return ArithmeticExpression(self, ArithmeticOperator.MOD, other)

    def __rmod__(self, other: Any) -> 'ArithmeticExpression':
        """Right modulo operator (%)."""
        return ArithmeticExpression(other, ArithmeticOperator.MOD, self)

    def __pow__(self, other: Any) -> 'ArithmeticExpression':
        """Power operator (^)."""
        return ArithmeticExpression(self, ArithmeticOperator.POW, other)

    def __rpow__(self, other: Any) -> 'ArithmeticExpression':
        """Right power operator (^)."""
        return ArithmeticExpression(other, ArithmeticOperator.POW, self)

    # ============================================================================
    # TEMPORAL OPERATORS FOR TEMPORAL EXPRESSIONS (NESTED OPERATIONS)
    # ============================================================================

    def date_add(self, value: Any) -> 'TemporalExpression':
        """Add to temporal expression result (for nested operations)."""
        if isinstance(value, int):
            return TemporalExpression(self, TemporalOperator.DATE_ADD_INT, value)
        else:
            return TemporalExpression(self, TemporalOperator.DATE_ADD_INTERVAL, value)

    def date_sub(self, value: Any) -> 'TemporalExpression':
        """Subtract from temporal expression result (for nested operations)."""
        from .kuzu_query_fields import QueryField

        if isinstance(value, QueryField):
            return TemporalExpression(self, TemporalOperator.DATE_SUB_DATE, value)
        elif isinstance(value, str) and 'INTERVAL' in value.upper():
            return TemporalExpression(self, TemporalOperator.DATE_SUB_INTERVAL, value)
        elif isinstance(value, str):
            return TemporalExpression(self, TemporalOperator.DATE_SUB_DATE, value)
        else:
            return TemporalExpression(self, TemporalOperator.DATE_SUB_INTERVAL, value)

    def timestamp_add(self, interval: Any) -> 'TemporalExpression':
        """Add interval to temporal expression result."""
        return TemporalExpression(self, TemporalOperator.TIMESTAMP_ADD_INTERVAL, interval)

    def timestamp_sub(self, value: Any) -> 'TemporalExpression':
        """Subtract from temporal expression result."""
        from .kuzu_query_fields import QueryField

        if isinstance(value, QueryField):
            return TemporalExpression(self, TemporalOperator.TIMESTAMP_SUB_TIMESTAMP, value)
        elif isinstance(value, str) and 'INTERVAL' in value.upper():
            return TemporalExpression(self, TemporalOperator.TIMESTAMP_SUB_INTERVAL, value)
        elif isinstance(value, str):
            return TemporalExpression(self, TemporalOperator.TIMESTAMP_SUB_TIMESTAMP, value)
        else:
            return TemporalExpression(self, TemporalOperator.TIMESTAMP_SUB_INTERVAL, value)

    def interval_add(self, value: Any) -> 'TemporalExpression':
        """Add to temporal expression result."""
        if isinstance(value, str):
            if 'DATE(' in value.upper() and 'TIMESTAMP(' not in value.upper():
                return TemporalExpression(self, TemporalOperator.INTERVAL_ADD_DATE, value)
            elif 'TIMESTAMP(' in value.upper():
                return TemporalExpression(self, TemporalOperator.INTERVAL_ADD_TIMESTAMP, value)
            elif 'INTERVAL(' in value.upper():
                return TemporalExpression(self, TemporalOperator.INTERVAL_ADD_INTERVAL, value)
            else:
                return TemporalExpression(self, TemporalOperator.INTERVAL_ADD_INTERVAL, value)
        else:
            return TemporalExpression(self, TemporalOperator.INTERVAL_ADD_INTERVAL, value)

    def interval_sub(self, interval: Any) -> 'TemporalExpression':
        """Subtract interval from temporal expression result."""
        return TemporalExpression(self, TemporalOperator.INTERVAL_SUB_INTERVAL, interval)


class PatternExpression(FilterExpression):
    """Pattern matching expression for regex operations."""

    _param_counter = 0

    def __init__(self, left: Any, operator: PatternOperator, pattern: str):
        self.left = left
        self.operator = operator
        self.pattern = pattern

        # Use deterministic parameter names without hash collisions
        PatternExpression._param_counter += 1
        self.param_id = PatternExpression._param_counter
        self.parameter_name = f"pattern_{self.param_id}"

    def to_cypher(self, alias_map: Dict[str, str], param_prefix: str = "", relationship_alias: Optional[str] = None, post_with: bool = False) -> str:
        """Convert to Cypher pattern expression."""
        left_expr = self._format_operand(self.left, alias_map, param_prefix, relationship_alias, post_with)

        if self.operator == PatternOperator.REGEX_MATCH:
            return f"{left_expr} =~ ${param_prefix}{self.parameter_name}"
        else:  # NOT_REGEX_MATCH
            return f"{left_expr} !~ ${param_prefix}{self.parameter_name}"

    def _format_operand(self, operand: Any, alias_map: Dict[str, str], param_prefix: str, relationship_alias: Optional[str], post_with: bool) -> str:
        """Format an operand for Cypher output."""
        from .kuzu_query_fields import QueryField

        if isinstance(operand, QueryField):
            # Handle QueryField operands
            parts = operand.field_path.split(".", 1)
            if len(parts) == 2:
                alias, field = parts
                mapped_alias = alias_map[alias] if alias in alias_map else (next(iter(alias_map.values())) if alias_map else alias)
                return f"{mapped_alias}.{field}"
            else:
                if post_with:
                    return operand.field_path
                elif relationship_alias:
                    return f"{relationship_alias}.{operand.field_path}"
                else:
                    default_alias = next(iter(alias_map.values())) if alias_map else "n"
                    return f"{default_alias}.{operand.field_path}"
        elif isinstance(operand, (ArithmeticExpression, TemporalExpression, FunctionExpression, PatternExpression)):
            # Handle nested expressions
            return operand.to_cypher(alias_map, param_prefix, relationship_alias, post_with)
        else:
            # PatternExpression only has one operand (left), so it must be self.left
            if operand is not self.left:
                raise ValueError(f"Operand '{operand}' of type {type(operand)} is not the left operand of PatternExpression")
            # For PatternExpression, non-field operands are not parameterized in _format_operand
            # They are handled directly in to_cypher method
            raise ValueError(f"Unexpected operand type {type(operand)} in PatternExpression._format_operand")

    def get_parameters(self) -> Dict[str, Any]:
        """Get query parameters."""
        params = {self.parameter_name: self.pattern}

        # Add parameters from nested expressions
        if isinstance(self.left, (ArithmeticExpression, TemporalExpression, FunctionExpression, PatternExpression)):
            params.update(self.left.get_parameters())

        return params

    def get_field_references(self) -> Set[str]:
        """Get field references."""
        from .kuzu_query_fields import QueryField
        refs = set()

        if isinstance(self.left, QueryField):
            refs.add(self.left.field_path)
        elif isinstance(self.left, (ArithmeticExpression, TemporalExpression, FunctionExpression, PatternExpression)):
            refs.update(self.left.get_field_references())

        return refs


class FunctionFilterExpression(FilterExpression):
    """Filter expression for function comparisons."""

    def __init__(
        self,
        function_expr: 'FunctionExpression',
        operator: ComparisonOperator,
        value: Any = None,
        parameter_name: Optional[str] = None
    ):
        self.function_expr = function_expr
        self.operator = operator
        self.value = value
        self.parameter_name = parameter_name or f"func_filter_{abs(hash((str(function_expr), operator, str(value))))}"

    def to_cypher(self, alias_map: Dict[str, str], param_prefix: str = "", relationship_alias: Optional[str] = None, post_with: bool = False) -> str:
        """Convert to Cypher WHERE clause fragment.

        Supports QueryField/FunctionExpression on the RHS without parameterization.
        """
        function_cypher = self.function_expr.to_cypher(alias_map, param_prefix, relationship_alias, post_with)

        # Resolve RHS when it's a field or function expression
        rhs_expr: Optional[str] = None
        from .kuzu_query_fields import QueryField as _QF
        if isinstance(self.value, _QF):
            parts = self.value.field_path.split(".", 1)
            if len(parts) == 2:
                alias, field = parts
                mapped = alias_map[alias] if alias in alias_map else (next(iter(alias_map.values())) if alias_map else alias)
                rhs_expr = f"{mapped}.{field}"
            else:
                if post_with:
                    rhs_expr = self.value.field_path
                elif relationship_alias:
                    rhs_expr = f"{relationship_alias}.{self.value.field_path}"
                else:
                    default_alias = next(iter(alias_map.values())) if alias_map else "n"
                    rhs_expr = f"{default_alias}.{self.value.field_path}"
        elif isinstance(self.value, (FunctionExpression, ArithmeticExpression, TemporalExpression, PatternExpression)):
            rhs_expr = self.value.to_cypher(alias_map, param_prefix, relationship_alias, post_with)

        param_name = f"{param_prefix}{self.parameter_name}"

        if self.operator == ComparisonOperator.IS_NULL:
            return f"{function_cypher} IS NULL"
        elif self.operator == ComparisonOperator.IS_NOT_NULL:
            return f"{function_cypher} IS NOT NULL"
        elif self.operator == ComparisonOperator.IN:
            return f"{function_cypher} IN ${param_name}"
        elif self.operator == ComparisonOperator.NOT_IN:
            return f"NOT {function_cypher} IN ${param_name}"
        else:
            operator_str = {
                ComparisonOperator.EQ: "=",
                ComparisonOperator.NEQ: "<>",
                ComparisonOperator.LT: "<",
                ComparisonOperator.LTE: "<=",
                ComparisonOperator.GT: ">",
                ComparisonOperator.GTE: ">=",
                ComparisonOperator.LIKE: "LIKE",
                ComparisonOperator.NOT_LIKE: "NOT LIKE",
                ComparisonOperator.REGEX: "=~",
                ComparisonOperator.NOT_REGEX: "!~",
                ComparisonOperator.REGEX_MATCH: "=~",
                ComparisonOperator.NOT_REGEX_MATCH: "!~",
            }.get(self.operator, "=")

            if rhs_expr is not None:
                return f"{function_cypher} {operator_str} {rhs_expr}"

            if isinstance(self.value, (int, float, bool)):
                value_str = str(self.value).lower() if isinstance(self.value, bool) else str(self.value)
                return f"{function_cypher} {operator_str} {value_str}"
            else:
                return f"{function_cypher} {operator_str} ${param_name}"

    def get_parameters(self) -> Dict[str, Any]:
        """Get query parameters."""
        # Start with parameters from LHS function
        params = self.function_expr.get_parameters()

        # Add parameters from RHS function if present
        from .kuzu_query_fields import QueryField as _QF
        if isinstance(self.value, (FunctionExpression, ArithmeticExpression, TemporalExpression, PatternExpression)):
            params.update(self.value.get_parameters())
        elif isinstance(self.value, _QF):
            # Field references are inlined, no parameter
            pass
        else:
            if not isinstance(self.value, (int, float, bool)) and self.operator not in (
                ComparisonOperator.IS_NULL, ComparisonOperator.IS_NOT_NULL
            ):
                params[self.parameter_name] = self.value
        return params

    def get_field_references(self) -> Set[str]:
        """Get field references."""
        return self.function_expr.get_field_references()


class FunctionExpression(FilterExpression):
    """Function expression for Kuzu functions."""

    _param_counter = 0

    def __init__(self, function_name: str, args: List[Any], alias: Optional[str] = None):
        self.function_name = function_name
        self.args = args
        self.alias = alias

        # Use deterministic parameter names without hash collisions
        FunctionExpression._param_counter += 1
        self.param_id = FunctionExpression._param_counter
        self.parameter_names = [f"func_arg_{self.param_id}_{i}" for i in range(len(args))]

    def to_cypher(self, alias_map: Dict[str, str], param_prefix: str = "", relationship_alias: Optional[str] = None, post_with: bool = False) -> str:
        """Convert to Cypher function call."""
        formatted_args = []

        for i, arg in enumerate(self.args):
            formatted_arg = self._format_argument(arg, alias_map, param_prefix, relationship_alias, post_with, i)
            formatted_args.append(formatted_arg)

        args_str = ", ".join(formatted_args)
        function_call = f"{self.function_name}({args_str})"

        if self.alias:
            function_call += f" AS {self.alias}"

        return function_call

    def _format_argument(self, arg: Any, alias_map: Dict[str, str], param_prefix: str, relationship_alias: Optional[str], post_with: bool, arg_index: int) -> str:
        """Format a function argument for Cypher output."""
        from .kuzu_query_fields import QueryField

        if isinstance(arg, QueryField):
            # Handle QueryField arguments
            parts = arg.field_path.split(".", 1)
            if len(parts) == 2:
                alias, field = parts
                mapped_alias = alias_map[alias] if alias in alias_map else (next(iter(alias_map.values())) if alias_map else alias)
                return f"{mapped_alias}.{field}"
            else:
                if post_with:
                    return arg.field_path
                elif relationship_alias:
                    return f"{relationship_alias}.{arg.field_path}"
                else:
                    default_alias = next(iter(alias_map.values())) if alias_map else "n"
                    return f"{default_alias}.{arg.field_path}"
        elif isinstance(arg, (ArithmeticExpression, FunctionExpression, TemporalExpression, PatternExpression)):
            # Handle nested expressions
            return arg.to_cypher(alias_map, param_prefix, relationship_alias, post_with)
        elif isinstance(arg, (int, float, bool)):
            # Handle literals directly
            return str(arg).lower() if isinstance(arg, bool) else str(arg)
        elif isinstance(arg, str):
            # Handle string literals - some functions need direct strings, others need parameters
            if self.function_name in ['array_slice', 'array_extract', 'substring', 'left', 'right']:
                return f"'{arg}'" if not arg.isdigit() else arg
            else:
                return f"${param_prefix}{self.parameter_names[arg_index]}"
        elif isinstance(arg, (list, tuple, dict)):
            # Handle collection arguments as parameters
            return f"${param_prefix}{self.parameter_names[arg_index]}"
        else:
            # Handle any other argument types as parameters
            return f"${param_prefix}{self.parameter_names[arg_index]}"

    def get_parameters(self) -> Dict[str, Any]:
        """Get query parameters."""
        params = {}

        from .kuzu_query_fields import QueryField

        for i, arg in enumerate(self.args):
            if not isinstance(arg, (int, float, bool)) and not isinstance(arg, QueryField):
                if not isinstance(arg, str) or self.function_name not in ['array_slice', 'array_extract', 'substring', 'left', 'right']:
                    params[self.parameter_names[i]] = arg

            # Add parameters from nested expressions
            if isinstance(arg, (ArithmeticExpression, TemporalExpression, FunctionExpression, PatternExpression)):
                params.update(arg.get_parameters())

        return params

    def get_field_references(self) -> Set[str]:
        """Get field references."""
        from .kuzu_query_fields import QueryField
        refs = set()

        for arg in self.args:
            if isinstance(arg, QueryField):
                refs.add(arg.field_path)
            elif isinstance(arg, (ArithmeticExpression, TemporalExpression, FunctionExpression, PatternExpression)):
                refs.update(arg.get_field_references())

        return refs

    # ============================================================================
    # ARITHMETIC OPERATORS FOR FUNCTION EXPRESSIONS
    # ============================================================================

    def __add__(self, other: Any) -> 'ArithmeticExpression':
        """Addition operator (+)."""
        return ArithmeticExpression(self, ArithmeticOperator.ADD, other)

    def __radd__(self, other: Any) -> 'ArithmeticExpression':
        """Right addition operator (+)."""
        return ArithmeticExpression(other, ArithmeticOperator.ADD, self)

    def __sub__(self, other: Any) -> 'ArithmeticExpression':
        """Subtraction operator (-)."""
        return ArithmeticExpression(self, ArithmeticOperator.SUB, other)

    def __rsub__(self, other: Any) -> 'ArithmeticExpression':
        """Right subtraction operator (-)."""
        return ArithmeticExpression(other, ArithmeticOperator.SUB, self)

    def __mul__(self, other: Any) -> 'ArithmeticExpression':
        """Multiplication operator (*)."""
        return ArithmeticExpression(self, ArithmeticOperator.MUL, other)

    def __rmul__(self, other: Any) -> 'ArithmeticExpression':
        """Right multiplication operator (*)."""
        return ArithmeticExpression(other, ArithmeticOperator.MUL, self)

    def __truediv__(self, other: Any) -> 'ArithmeticExpression':
        """Division operator (/)."""
        return ArithmeticExpression(self, ArithmeticOperator.DIV, other)

    def __rtruediv__(self, other: Any) -> 'ArithmeticExpression':
        """Right division operator (/)."""
        return ArithmeticExpression(other, ArithmeticOperator.DIV, self)

    def __mod__(self, other: Any) -> 'ArithmeticExpression':
        """Modulo operator (%)."""
        return ArithmeticExpression(self, ArithmeticOperator.MOD, other)

    def __rmod__(self, other: Any) -> 'ArithmeticExpression':
        """Right modulo operator (%)."""
        return ArithmeticExpression(other, ArithmeticOperator.MOD, self)

    def __pow__(self, other: Any) -> 'ArithmeticExpression':
        """Power operator (^)."""
        return ArithmeticExpression(self, ArithmeticOperator.POW, other)

    def __rpow__(self, other: Any) -> 'ArithmeticExpression':
        """Right power operator (^)."""
        return ArithmeticExpression(other, ArithmeticOperator.POW, self)

    # ============================================================================
    # PATTERN MATCHING METHODS FOR FUNCTION EXPRESSIONS
    # ============================================================================

    def regex_match(self, pattern: str) -> 'PatternExpression':
        """Regex match operator (=~) for function expressions."""
        return PatternExpression(self, PatternOperator.REGEX_MATCH, pattern)

    def not_regex_match(self, pattern: str) -> 'PatternExpression':
        """Negative regex match operator (!~) for function expressions."""
        return PatternExpression(self, PatternOperator.NOT_REGEX_MATCH, pattern)

    def __matmul__(self, pattern: str) -> 'PatternExpression':
        """Pattern matching operator (=~) using @ symbol for function expressions."""
        return PatternExpression(self, PatternOperator.REGEX_MATCH, pattern)

    def __rmatmul__(self, pattern: str) -> 'PatternExpression':
        """Right pattern matching operator (=~) using @ symbol for function expressions."""
        return PatternExpression(self, PatternOperator.REGEX_MATCH, pattern)

    # ============================================================================
    # TEXT FUNCTIONS FOR FUNCTION EXPRESSIONS (CHAINING SUPPORT)
    # ============================================================================

    def upper(self) -> 'FunctionExpression':
        """Convert to uppercase."""
        return FunctionExpression("upper", [self])

    def lower(self) -> 'FunctionExpression':
        """Convert to lowercase."""
        return FunctionExpression("lower", [self])

    def substring(self, start: int, length: int) -> 'FunctionExpression':
        """Extract substring."""
        return FunctionExpression("substring", [self, start, length])

    def trim(self) -> 'FunctionExpression':
        """Trim whitespace."""
        return FunctionExpression("trim", [self])

    def size(self) -> 'FunctionExpression':
        """Get size/length."""
        return FunctionExpression("size", [self])

    def concat(self, *args) -> 'FunctionExpression':
        """Concatenate with other values."""
        return FunctionExpression("concat", [self] + list(args))

    # ============================================================================
    # COMPARISON OPERATORS FOR FUNCTION EXPRESSIONS
    # ============================================================================

    def __eq__(self, other) -> 'FunctionFilterExpression':
        """Equality comparison."""
        return FunctionFilterExpression(self, ComparisonOperator.EQ, other)

    def __ne__(self, other) -> 'FunctionFilterExpression':
        """Not equal comparison."""
        return FunctionFilterExpression(self, ComparisonOperator.NEQ, other)

    def __lt__(self, other) -> 'FunctionFilterExpression':
        """Less than comparison."""
        return FunctionFilterExpression(self, ComparisonOperator.LT, other)

    def __le__(self, other) -> 'FunctionFilterExpression':
        """Less than or equal comparison."""
        return FunctionFilterExpression(self, ComparisonOperator.LTE, other)

    def __gt__(self, other) -> 'FunctionFilterExpression':
        """Greater than comparison."""
        return FunctionFilterExpression(self, ComparisonOperator.GT, other)

    def __ge__(self, other) -> 'FunctionFilterExpression':
        """Greater than or equal comparison."""
        return FunctionFilterExpression(self, ComparisonOperator.GTE, other)


class CastExpression(FilterExpression):
    """CAST expression for type conversion."""

    _param_counter = 0

    def __init__(self, value: Any, target_type: str, use_as_syntax: bool = False):
        self.value = value
        self.target_type = target_type
        self.use_as_syntax = use_as_syntax

        # Use deterministic parameter names without hash collisions
        CastExpression._param_counter += 1
        self.param_id = CastExpression._param_counter
        self.parameter_name = f"cast_value_{self.param_id}"

    def to_cypher(self, alias_map: Dict[str, str], param_prefix: str = "", relationship_alias: Optional[str] = None, post_with: bool = False) -> str:
        """Convert to Cypher CAST expression."""
        value_expr = self._format_value(self.value, alias_map, param_prefix, relationship_alias, post_with)

        if self.use_as_syntax:
            return f"CAST({value_expr} AS {self.target_type})"
        else:
            return f"CAST({value_expr}, \"{self.target_type}\")"

    def _format_value(self, value: Any, alias_map: Dict[str, str], param_prefix: str, relationship_alias: Optional[str], post_with: bool) -> str:
        """Format the value for Cypher output."""
        from .kuzu_query_fields import QueryField

        if isinstance(value, QueryField):
            # Handle QueryField values
            parts = value.field_path.split(".", 1)
            if len(parts) == 2:
                alias, field = parts
                mapped_alias = alias_map[alias] if alias in alias_map else (next(iter(alias_map.values())) if alias_map else alias)
                return f"{mapped_alias}.{field}"
            else:
                if post_with:
                    return value.field_path
                elif relationship_alias:
                    return f"{relationship_alias}.{value.field_path}"
                else:
                    default_alias = next(iter(alias_map.values())) if alias_map else "n"
                    return f"{default_alias}.{value.field_path}"
        elif isinstance(value, (ArithmeticExpression, FunctionExpression, TemporalExpression, PatternExpression, CastExpression)):
            # Handle nested expressions
            return value.to_cypher(alias_map, param_prefix, relationship_alias, post_with)
        elif isinstance(value, (int, float, bool)):
            # Handle literals directly
            return str(value).lower() if isinstance(value, bool) else str(value)
        else:
            # Handle parameters
            return f"${param_prefix}{self.parameter_name}"

    def get_parameters(self) -> Dict[str, Any]:
        """Get query parameters."""
        params = {}

        from .kuzu_query_fields import QueryField

        if not isinstance(self.value, (int, float, bool)) and not isinstance(self.value, QueryField):
            params[self.parameter_name] = self.value

        # Add parameters from nested expressions
        if isinstance(self.value, (ArithmeticExpression, TemporalExpression, FunctionExpression, PatternExpression, CastExpression)):
            params.update(self.value.get_parameters())

        return params

    def get_field_references(self) -> Set[str]:
        """Get field references."""
        from .kuzu_query_fields import QueryField
        refs = set()

        if isinstance(self.value, QueryField):
            refs.add(self.value.field_path)
        elif isinstance(self.value, (ArithmeticExpression, TemporalExpression, FunctionExpression, PatternExpression, CastExpression)):
            refs.update(self.value.get_field_references())

        return refs


class CaseExpression(FilterExpression):
    """CASE expression for conditional logic."""

    _param_counter = 0

    def __init__(self, input_expr: Any = None):
        self.input_expr = input_expr  # For simple form
        self.when_clauses: List[Tuple[Any, Any]] = []  # (condition, result) pairs
        self.else_clause: Any = None

        # Use deterministic parameter names without hash collisions
        CaseExpression._param_counter += 1
        self.param_id = CaseExpression._param_counter

    def when(self, condition: Any, result: Any) -> 'CaseExpression':
        """Add WHEN clause."""
        self.when_clauses.append((condition, result))
        return self

    def else_(self, result: Any) -> 'CaseExpression':
        """Add ELSE clause."""
        self.else_clause = result
        return self

    def to_cypher(self, alias_map: Dict[str, str], param_prefix: str = "", relationship_alias: Optional[str] = None, post_with: bool = False) -> str:
        """Convert to Cypher CASE expression."""
        parts = ["CASE"]

        # Add input expression for simple form
        if self.input_expr is not None:
            input_expr = self._format_operand(self.input_expr, alias_map, param_prefix, relationship_alias, post_with)
            parts.append(input_expr)

        # Add WHEN clauses
        for condition, result in self.when_clauses:
            condition_expr = self._format_operand(condition, alias_map, param_prefix, relationship_alias, post_with)
            result_expr = self._format_operand(result, alias_map, param_prefix, relationship_alias, post_with)
            parts.append(f"WHEN {condition_expr} THEN {result_expr}")

        # Add ELSE clause
        if self.else_clause is not None:
            else_expr = self._format_operand(self.else_clause, alias_map, param_prefix, relationship_alias, post_with)
            parts.append(f"ELSE {else_expr}")

        parts.append("END")
        return " ".join(parts)

    def _format_operand(self, operand: Any, alias_map: Dict[str, str], param_prefix: str, relationship_alias: Optional[str], post_with: bool) -> str:
        """Format an operand for Cypher output."""
        from .kuzu_query_fields import QueryField

        if isinstance(operand, QueryField):
            # Handle QueryField operands
            parts = operand.field_path.split(".", 1)
            if len(parts) == 2:
                alias, field = parts
                mapped_alias = alias_map[alias] if alias in alias_map else (next(iter(alias_map.values())) if alias_map else alias)
                return f"{mapped_alias}.{field}"
            else:
                if post_with:
                    return operand.field_path
                elif relationship_alias:
                    return f"{relationship_alias}.{operand.field_path}"
                else:
                    default_alias = next(iter(alias_map.values())) if alias_map else "n"
                    return f"{default_alias}.{operand.field_path}"
        elif isinstance(operand, (ArithmeticExpression, FunctionExpression, TemporalExpression, PatternExpression, CastExpression, CaseExpression)):
            # Handle nested expressions
            return operand.to_cypher(alias_map, param_prefix, relationship_alias, post_with)
        elif isinstance(operand, FilterExpression):
            # Handle filter expressions (for conditions)
            return operand.to_cypher(alias_map, param_prefix, relationship_alias, post_with)
        elif isinstance(operand, (int, float, bool)):
            # Handle literals directly
            return str(operand).lower() if isinstance(operand, bool) else str(operand)
        elif isinstance(operand, str):
            # Handle string literals
            return f"'{operand}'"
        else:
            # Handle parameters
            param_name = f"case_param_{self.param_id}_{len(self.when_clauses)}"
            return f"${param_prefix}{param_name}"

    def get_parameters(self) -> Dict[str, Any]:
        """Get query parameters."""
        params = {}

        from .kuzu_query_fields import QueryField

        # Add parameters from input expression
        if self.input_expr is not None and isinstance(self.input_expr, (ArithmeticExpression, TemporalExpression, FunctionExpression, PatternExpression, CastExpression, CaseExpression)):
            params.update(self.input_expr.get_parameters())

        # Add parameters from WHEN clauses
        for condition, result in self.when_clauses:
            if isinstance(condition, (ArithmeticExpression, TemporalExpression, FunctionExpression, PatternExpression, CastExpression, CaseExpression, FilterExpression)):
                params.update(condition.get_parameters())
            if isinstance(result, (ArithmeticExpression, TemporalExpression, FunctionExpression, PatternExpression, CastExpression, CaseExpression)):
                params.update(result.get_parameters())

        # Add parameters from ELSE clause
        if self.else_clause is not None and isinstance(self.else_clause, (ArithmeticExpression, TemporalExpression, FunctionExpression, PatternExpression, CastExpression, CaseExpression)):
            params.update(self.else_clause.get_parameters())

        return params

    def get_field_references(self) -> Set[str]:
        """Get field references."""
        from .kuzu_query_fields import QueryField
        refs = set()

        # Add references from input expression
        if isinstance(self.input_expr, QueryField):
            refs.add(self.input_expr.field_path)
        elif isinstance(self.input_expr, (ArithmeticExpression, TemporalExpression, FunctionExpression, PatternExpression, CastExpression, CaseExpression)):
            refs.update(self.input_expr.get_field_references())

        # Add references from WHEN clauses
        for condition, result in self.when_clauses:
            if isinstance(condition, QueryField):
                refs.add(condition.field_path)
            elif isinstance(condition, (ArithmeticExpression, TemporalExpression, FunctionExpression, PatternExpression, CastExpression, CaseExpression, FilterExpression)):
                refs.update(condition.get_field_references())

            if isinstance(result, QueryField):
                refs.add(result.field_path)
            elif isinstance(result, (ArithmeticExpression, TemporalExpression, FunctionExpression, PatternExpression, CastExpression, CaseExpression)):
                refs.update(result.get_field_references())

        # Add references from ELSE clause
        if isinstance(self.else_clause, QueryField):
            refs.add(self.else_clause.field_path)
        elif isinstance(self.else_clause, (ArithmeticExpression, TemporalExpression, FunctionExpression, PatternExpression, CastExpression, CaseExpression)):
            refs.update(self.else_clause.get_field_references())

        return refs

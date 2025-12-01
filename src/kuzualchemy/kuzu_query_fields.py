# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
Field descriptors and query field helpers for Kuzu query system.
"""

from typing import Any, Type, Optional, List, Tuple
from .kuzu_query_expressions import (
    ArithmeticExpression, FilterExpression, FieldFilterExpression, BetweenExpression,
    ComparisonOperator, FunctionExpression, OrderDirection, TemporalExpression,
    CastExpression, CaseExpression
)
from .constants import QueryFieldConstants, ValidationMessageConstants


class QueryField:
    """Field descriptor for building queries with operator overloading."""
    
    def __init__(self, field_name: str, model_class: Optional[Type[Any]] = None):
        self.field_name = field_name
        self.model_class = model_class

        # Include model class hint in field_path if provided; the builder maps model name -> alias
        if model_class is not None:
            self.field_path = f"{model_class.__name__}.{field_name}"
        else:
            self.field_path = field_name

    def __eq__(self, value: Any) -> FilterExpression:
        return FieldFilterExpression(self.field_path, ComparisonOperator.EQ, value)
    
    def __ne__(self, value: Any) -> FilterExpression:
        return FieldFilterExpression(self.field_path, ComparisonOperator.NEQ, value)
    
    def __lt__(self, value: Any) -> FilterExpression:
        return FieldFilterExpression(self.field_path, ComparisonOperator.LT, value)
    
    def __le__(self, value: Any) -> FilterExpression:
        return FieldFilterExpression(self.field_path, ComparisonOperator.LTE, value)
    
    def __gt__(self, value: Any) -> FilterExpression:
        return FieldFilterExpression(self.field_path, ComparisonOperator.GT, value)
    
    def __ge__(self, value: Any) -> FilterExpression:
        return FieldFilterExpression(self.field_path, ComparisonOperator.GTE, value)
    
    def in_(self, values: List[Any]) -> FilterExpression:
        """Check if field value is in list."""
        return FieldFilterExpression(self.field_path, ComparisonOperator.IN, values)
    
    def not_in(self, values: List[Any]) -> FilterExpression:
        """Check if field value is not in list."""
        return FieldFilterExpression(self.field_path, ComparisonOperator.NOT_IN, values)
    
    def like(self, pattern: str, case_sensitive: bool = True) -> FilterExpression:
        """Pattern matching with regex."""
        return FieldFilterExpression(
            self.field_path, ComparisonOperator.LIKE, pattern,
            case_sensitive=case_sensitive
        )
    
    def not_like(self, pattern: str, case_sensitive: bool = True) -> FilterExpression:
        """Negative pattern matching."""
        return FieldFilterExpression(
            self.field_path, ComparisonOperator.NOT_LIKE, pattern,
            case_sensitive=case_sensitive
        )
    
    def regex(self, pattern: str) -> FilterExpression:
        """Regex pattern matching."""
        return FieldFilterExpression(self.field_path, ComparisonOperator.REGEX, pattern)
    
    def not_regex(self, pattern: str) -> FilterExpression:
        """Negative regex pattern matching."""
        return FieldFilterExpression(self.field_path, ComparisonOperator.NOT_REGEX, pattern)
    
    def is_null(self) -> FilterExpression:
        """Check if field is NULL."""
        return FieldFilterExpression(self.field_path, ComparisonOperator.IS_NULL, None)
    
    def is_not_null(self) -> FilterExpression:
        """Check if field is not NULL."""
        return FieldFilterExpression(self.field_path, ComparisonOperator.IS_NOT_NULL, None)
    
    def between(self, start: Any, end: Any, inclusive: bool = True) -> FilterExpression:
        """Check if field value is between two values."""
        return BetweenExpression(self.field_path, start, end, inclusive)
    
    def contains_filter(self, value: Any) -> FilterExpression:
        """Check if field contains value (for lists/arrays) - filter operation."""
        return FieldFilterExpression(self.field_path, ComparisonOperator.CONTAINS, value)

    def starts_with_filter(self, value: str, case_sensitive: bool = True) -> FilterExpression:
        """Check if string field starts with value - filter operation."""
        return FieldFilterExpression(
            self.field_path, ComparisonOperator.STARTS_WITH, value,
            case_sensitive=case_sensitive
        )

    def ends_with_filter(self, value: str, case_sensitive: bool = True) -> FilterExpression:
        """Check if string field ends with value - filter operation."""
        return FieldFilterExpression(
            self.field_path, ComparisonOperator.ENDS_WITH, value,
            case_sensitive=case_sensitive
        )
    
    def exists(self) -> FilterExpression:
        """Check if property/relationship exists."""
        return FieldFilterExpression(self.field_path, ComparisonOperator.EXISTS, None)
    
    def not_exists(self) -> FilterExpression:
        """Check if property/relationship doesn't exist."""
        return FieldFilterExpression(self.field_path, ComparisonOperator.NOT_EXISTS, None)
    
    def asc(self) -> Tuple[str, OrderDirection]:
        """Create ascending order specification."""
        return (self.field_name, OrderDirection.ASC)
    
    def desc(self) -> Tuple[str, OrderDirection]:
        """Create descending order specification."""
        return (self.field_name, OrderDirection.DESC)

    # ============================================================================
    # TEXT FUNCTIONS - MISSING FUNCTIONALITY IMPLEMENTATION
    # ============================================================================

    def concat(self, *args: Any) -> 'FunctionExpression':
        """Concatenate multiple strings."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("concat", [self] + list(args))

    def lower(self) -> 'FunctionExpression':
        """Convert string to lowercase."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("lower", [self])

    def upper(self) -> 'FunctionExpression':
        """Convert string to uppercase."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("upper", [self])

    # initcap method moved to section above

    # Duplicate methods removed - implementations are in the section above

    def substring(self, start: int, length: int) -> 'FunctionExpression':
        """Extract substring from start position with given length (1-based index)."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("substring", [self, start, length])

    def size(self) -> 'FunctionExpression':
        """Get the number of characters in string."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("size", [self])

    def trim(self) -> 'FunctionExpression':
        """Remove whitespace from both ends of string."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("trim", [self])

    def contains(self, substring: Any) -> 'FunctionExpression':
        """Returns true if substring is contained in this string."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("contains", [self, substring])

    def ends_with(self, suffix: Any) -> 'FunctionExpression':
        """Returns whether string ends with suffix."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("ends_with", [self, suffix])

    def lcase(self) -> 'FunctionExpression':
        """Alias of lower - returns the string in lower case."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("lcase", [self])

    def left(self, count: int) -> 'FunctionExpression':
        """Returns the leftmost count number of characters from string."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("left", [self, count])

    def levenshtein(self, other: Any) -> 'FunctionExpression':
        """Returns the minimum number of single-character edits required to transform strings."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("levenshtein", [self, other])

    def lpad(self, count: int, character: str) -> 'FunctionExpression':
        """Pads the string with the character from the left until it has count characters."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("lpad", [self, count, character])

    def ltrim(self) -> 'FunctionExpression':
        """Removes any whitespace in the beginning of the string."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("ltrim", [self])

    def repeat(self, count: int) -> 'FunctionExpression':
        """Repeats the string count number of times."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("repeat", [self, count])

    def reverse(self) -> 'FunctionExpression':
        """Reverses the string."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("reverse", [self])

    def right(self, count: int) -> 'FunctionExpression':
        """Returns the rightmost count number of characters from string."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("right", [self, count])

    def rpad(self, count: int, character: str) -> 'FunctionExpression':
        """Pads the string with the character from the right until it has count characters."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("rpad", [self, count, character])

    def rtrim(self) -> 'FunctionExpression':
        """Removes any whitespace in the end of the string."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("rtrim", [self])

    def starts_with(self, prefix: Any) -> 'FunctionExpression':
        """Returns whether string starts with prefix."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("starts_with", [self, prefix])

    def substr(self, start: int, length: int) -> 'FunctionExpression':
        """Alias of substring."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("substr", [self, start, length])

    def ucase(self) -> 'FunctionExpression':
        """Alias of upper - returns the string in upper case."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("ucase", [self])

    def initcap(self) -> 'FunctionExpression':
        """Returns the string with only the first letter in uppercase."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("initcap", [self])

    # Duplicate methods removed - implementations are in the section above

    def string_split(self, separator: str) -> 'FunctionExpression':
        """Split string along separator into a list."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("string_split", [self, separator])

    def split_part(self, separator: str, index: int) -> 'FunctionExpression':
        """Split string along separator and return part at index (1-based)."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("split_part", [self, separator, index])

    # ============================================================================
    # PATTERN MATCHING FUNCTIONS - MISSING FUNCTIONALITY IMPLEMENTATION
    # ============================================================================

    def regexp_matches(self, pattern: str) -> 'FunctionExpression':
        """Check if string matches regular expression pattern."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("regexp_matches", [self, pattern])

    def regexp_replace(
        self, pattern: str, replacement: str, options: Optional[str] = None
    ) -> 'FunctionExpression':
        """Replace the matching part of this field with replacement.

        Args:
            pattern: The regular expression pattern to match
            replacement: The replacement string
            options: Optional flags (e.g., 'g' for global replacement)

        Returns:
            FunctionExpression that evaluates to the modified string

        Example:
            field.regexp_replace('b.b', 'a') -> replaces pattern with 'a'
            field.regexp_replace('\\s+', '', 'g') -> removes all whitespace globally
        """
        from .kuzu_query_expressions import FunctionExpression
        if options is not None:
            return FunctionExpression("regexp_replace", [self, pattern, replacement, options])
        else:
            return FunctionExpression("regexp_replace", [self, pattern, replacement])

    def regexp_extract(self, pattern: str, group: int = 0) -> 'FunctionExpression':
        """Extract substring matching pattern (optionally from specific group)."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("regexp_extract", [self, pattern, group])

    def regexp_extract_all(self, pattern: str, group: int = 0) -> 'FunctionExpression':
        """Extract all substrings matching pattern (optionally from specific group)."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("regexp_extract_all", [self, pattern, group])

    def regexp_split_to_array(
        self, pattern: str, options: Optional[str] = None
    ) -> 'FunctionExpression':
        """Split this field along the regex and extract all occurrences between regex.

        Args:
            pattern: The regular expression pattern to split on
            options: Optional flags for splitting behavior

        Returns:
            FunctionExpression that evaluates to a list of split strings

        Example:
            field.regexp_split_to_array(' ') -> splits on spaces
        """
        from .kuzu_query_expressions import FunctionExpression
        if options is not None:
            return FunctionExpression("regexp_split_to_array", [self, pattern, options])
        else:
            return FunctionExpression("regexp_split_to_array", [self, pattern])

    # ============================================================================
    # LIST FUNCTIONS - MISSING FUNCTIONALITY IMPLEMENTATION
    # ============================================================================

    def list_concat(self, other: Any) -> 'FunctionExpression':
        """Concatenate two lists."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("list_concat", [self, other])

    def list_cat(self, other: Any) -> 'FunctionExpression':
        """Concatenate two lists (alias for list_concat)."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("list_cat", [self, other])

    def array_concat(self, other: Any) -> 'FunctionExpression':
        """Concatenate two arrays (alias for list_concat)."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("array_concat", [self, other])

    def array_cat(self, other: Any) -> 'FunctionExpression':
        """Concatenate two arrays (alias for list_concat)."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("array_cat", [self, other])

    def list_append(self, element: Any) -> 'FunctionExpression':
        """Append element to list."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("list_append", [self, element])

    def array_append(self, element: Any) -> 'FunctionExpression':
        """Append element to array (alias for list_append)."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("array_append", [self, element])

    def array_push_back(self, element: Any) -> 'FunctionExpression':
        """Append element to array (alias for list_append)."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("array_push_back", [self, element])

    def list_prepend(self, element: Any) -> 'FunctionExpression':
        """Prepend element to list."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("list_prepend", [self, element])

    def array_prepend(self, element: Any) -> 'FunctionExpression':
        """Prepend element to array (alias for list_prepend)."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("array_prepend", [self, element])

    def array_push_front(self, element: Any) -> 'FunctionExpression':
        """Prepend element to array (alias for list_prepend)."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("array_push_front", [self, element])

    def list_position(self, element: Any) -> 'FunctionExpression':
        """Get position of element in list (1-based index)."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("list_position", [self, element])

    def list_indexof(self, element: Any) -> 'FunctionExpression':
        """Get position of element in list (alias for list_position)."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("list_indexof", [self, element])

    def array_position(self, element: Any) -> 'FunctionExpression':
        """Get position of element in array (alias for list_position)."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("array_position", [self, element])

    def array_indexof(self, element: Any) -> 'FunctionExpression':
        """Get position of element in array (alias for list_position)."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("array_indexof", [self, element])

    def list_contains(self, element: Any) -> 'FunctionExpression':
        """Check if list contains element."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("list_contains", [self, element])

    def list_has(self, element: Any) -> 'FunctionExpression':
        """Check if list contains element (alias for list_contains)."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("list_has", [self, element])

    def array_contains(self, element: Any) -> 'FunctionExpression':
        """Check if array contains element (alias for list_contains)."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("array_contains", [self, element])

    def array_has(self, element: Any) -> 'FunctionExpression':
        """Check if array contains element (alias for list_contains)."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("array_has", [self, element])

    def list_slice(self, begin: int, end: int) -> 'FunctionExpression':
        """Extract sub-list using slice conventions."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("list_slice", [self, begin, end])

    def list_reverse(self) -> 'FunctionExpression':
        """Reverse list elements."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("list_reverse", [self])

    def list_sort(self, order: str = "ASC") -> 'FunctionExpression':
        """Sort list elements."""
        from .kuzu_query_expressions import FunctionExpression
        if order.upper() == "ASC":
            return FunctionExpression("list_sort", [self])
        else:
            return FunctionExpression("list_sort", [self, order])

    def list_reverse_sort(self) -> 'FunctionExpression':
        """Sort list elements in descending order."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("list_reverse_sort", [self])

    def list_sum(self) -> 'FunctionExpression':
        """Sum elements of the list."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("list_sum", [self])

    def list_product(self) -> 'FunctionExpression':
        """Multiply elements of the list."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("list_product", [self])

    def list_distinct(self) -> 'FunctionExpression':
        """Remove NULLs and duplicate values from list."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("list_distinct", [self])

    def list_unique(self) -> 'FunctionExpression':
        """Count number of unique elements in list."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("list_unique", [self])

    def list_any_value(self) -> 'FunctionExpression':
        """Return first non-NULL value from list."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("list_any_value", [self])

    def list_to_string(self, separator: str) -> 'FunctionExpression':
        """Convert list to string with separator."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("list_to_string", [separator, self])

    # ============================================================================
    # NUMERIC FUNCTIONS - MISSING FUNCTIONALITY IMPLEMENTATION
    # ============================================================================

    def abs(self) -> 'FunctionExpression':
        """Return absolute value."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("abs", [self])

    def acos(self) -> 'FunctionExpression':
        """Return arccosine."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("acos", [self])

    def asin(self) -> 'FunctionExpression':
        """Return arcsine."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("asin", [self])

    def atan(self) -> 'FunctionExpression':
        """Return arctangent."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("atan", [self])

    def atan2(self, y: Any) -> 'FunctionExpression':
        """Return arctangent of x, y."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("atan2", [self, y])

    def bitwise_xor(self, other: Any) -> 'FunctionExpression':
        """Return bitwise XOR."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("bitwise_xor", [self, other])

    def ceil(self) -> 'FunctionExpression':
        """Round up to next integer."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("ceil", [self])

    def ceiling(self) -> 'FunctionExpression':
        """Round up to next integer (alias for ceil)."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("ceiling", [self])

    def cos(self) -> 'FunctionExpression':
        """Return cosine."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("cos", [self])

    def cot(self) -> 'FunctionExpression':
        """Return cotangent."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("cot", [self])

    def degrees(self) -> 'FunctionExpression':
        """Convert radians to degrees."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("degrees", [self])

    def even(self) -> 'FunctionExpression':
        """Round to next even number."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("even", [self])

    def factorial(self) -> 'FunctionExpression':
        """Return factorial."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("factorial", [self])

    def floor(self) -> 'FunctionExpression':
        """Round down to nearest integer."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("floor", [self])

    def gamma(self) -> 'FunctionExpression':
        """Return gamma function."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("gamma", [self])

    def lgamma(self) -> 'FunctionExpression':
        """Return log of gamma function."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("lgamma", [self])

    def ln(self) -> 'FunctionExpression':
        """Return natural logarithm."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("ln", [self])

    def log(self) -> 'FunctionExpression':
        """Return base-10 logarithm."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("log", [self])

    def log2(self) -> 'FunctionExpression':
        """Return base-2 logarithm."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("log2", [self])

    def log10(self) -> 'FunctionExpression':
        """Return base-10 logarithm (alias for log)."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("log10", [self])

    def negate(self) -> 'FunctionExpression':
        """Return negative value."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("negate", [self])

    def pow(self, exponent: Any) -> 'FunctionExpression':
        """Return value raised to power."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("pow", [self, exponent])

    def radians(self) -> 'FunctionExpression':
        """Convert degrees to radians."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("radians", [self])

    def round(self, precision: int = 0) -> 'FunctionExpression':
        """Round to specified decimal places."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("round", [self, precision])

    def sin(self) -> 'FunctionExpression':
        """Return sine."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("sin", [self])

    def sign(self) -> 'FunctionExpression':
        """Return sign (-1, 0, or 1)."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("sign", [self])

    def sqrt(self) -> 'FunctionExpression':
        """Return square root."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("sqrt", [self])

    def tan(self) -> 'FunctionExpression':
        """Return tangent."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("tan", [self])

    # ============================================================================
    # DATE FUNCTIONS - MISSING FUNCTIONALITY IMPLEMENTATION
    # ============================================================================

    def date_part(self, part: str) -> 'FunctionExpression':
        """Extract part from date."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("date_part", [part, self])

    def datepart(self, part: str) -> 'FunctionExpression':
        """Extract part from date (alias for date_part)."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("datepart", [part, self])

    def date_trunc(self, part: str) -> 'FunctionExpression':
        """Truncate date to specified precision."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("date_trunc", [part, self])

    def datetrunc(self, part: str) -> 'FunctionExpression':
        """Truncate date to specified precision (alias for date_trunc)."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("datetrunc", [part, self])

    def dayname(self) -> 'FunctionExpression':
        """Get English name of the day."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("dayname", [self])

    def monthname(self) -> 'FunctionExpression':
        """Get English name of the month."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("monthname", [self])

    def last_day(self) -> 'FunctionExpression':
        """Get last day of the month."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("last_day", [self])

    def greatest(self, other: Any) -> 'FunctionExpression':
        """Return the later of two dates."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("greatest", [self, other])

    def least(self, other: Any) -> 'FunctionExpression':
        """Return the earlier of two dates."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("least", [self, other])

    # ============================================================================
    # TIMESTAMP FUNCTIONS - MISSING FUNCTIONALITY IMPLEMENTATION
    # ============================================================================

    def century(self) -> 'FunctionExpression':
        """Get century from timestamp."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("century", [self])

    def epoch_ms(self) -> 'FunctionExpression':
        """Convert milliseconds to timestamp."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("epoch_ms", [self])

    def to_epoch_ms(self) -> 'FunctionExpression':
        """Convert timestamp to milliseconds."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("to_epoch_ms", [self])

    def to_timestamp(self) -> 'FunctionExpression':
        """Convert epoch seconds to timestamp."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("to_timestamp", [self])

    # ============================================================================
    # INTERVAL FUNCTIONS - MISSING FUNCTIONALITY IMPLEMENTATION
    # ============================================================================

    def to_years(self) -> 'FunctionExpression':
        """Convert integer to year interval."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("to_years", [self])

    def to_months(self) -> 'FunctionExpression':
        """Convert integer to month interval."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("to_months", [self])

    def to_days(self) -> 'FunctionExpression':
        """Convert integer to day interval."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("to_days", [self])

    def to_hours(self) -> 'FunctionExpression':
        """Convert integer to hour interval."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("to_hours", [self])

    def to_minutes(self) -> 'FunctionExpression':
        """Convert integer to minute interval."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("to_minutes", [self])

    def to_seconds(self) -> 'FunctionExpression':
        """Convert integer to second interval."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("to_seconds", [self])

    def to_milliseconds(self) -> 'FunctionExpression':
        """Convert integer to millisecond interval."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("to_milliseconds", [self])

    def to_microseconds(self) -> 'FunctionExpression':
        """Convert integer to microsecond interval."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("to_microseconds", [self])

    # ============================================================================
    # ARRAY FUNCTIONS - MISSING FUNCTIONALITY IMPLEMENTATION
    # ============================================================================

    def array_distance(self, other: Any) -> 'FunctionExpression':
        """Calculate Euclidean distance between two arrays."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("array_distance", [self, other])

    def array_squared_distance(self, other: Any) -> 'FunctionExpression':
        """Calculate squared Euclidean distance between two arrays."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("array_squared_distance", [self, other])

    def array_dot_product(self, other: Any) -> 'FunctionExpression':
        """Calculate dot product of two arrays."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("array_dot_product", [self, other])

    def array_inner_product(self, other: Any) -> 'FunctionExpression':
        """Calculate inner product of two arrays (alias for dot product)."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("array_inner_product", [self, other])

    def array_cross_product(self, other: Any) -> 'FunctionExpression':
        """Calculate cross product of two arrays."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("array_cross_product", [self, other])

    def array_cosine_similarity(self, other: Any) -> 'FunctionExpression':
        """Calculate cosine similarity of two arrays."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("array_cosine_similarity", [self, other])

    # ============================================================================
    # UTILITY FUNCTIONS - MISSING FUNCTIONALITY IMPLEMENTATION
    # ============================================================================

    def coalesce(self, *args: Any) -> 'FunctionExpression':
        """Return first non-NULL value from arguments."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("coalesce", [self] + list(args))

    def ifnull(self, alternative: Any) -> 'FunctionExpression':
        """Two-argument version of coalesce."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("ifnull", [self, alternative])

    def nullif(self, compare_value: Any) -> 'FunctionExpression':
        """Return NULL if values are equal, otherwise return first value."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("nullif", [self, compare_value])

    def constant_or_null(self, check_value: Any) -> 'FunctionExpression':
        """Return first value if second is not NULL, otherwise return NULL."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("constant_or_null", [self, check_value])

    def count_if(self) -> 'FunctionExpression':
        """Return 1 if value is true or non-zero, otherwise 0."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("count_if", [self])

    def typeof(self) -> 'FunctionExpression':
        """Return the data type name of the value."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("typeof", [self])

    def error(self) -> 'FunctionExpression':
        """Throw value as a runtime exception."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("error", [self])

    # ============================================================================
    # HASH FUNCTIONS - MISSING FUNCTIONALITY IMPLEMENTATION
    # ============================================================================

    def md5(self) -> 'FunctionExpression':
        """Return MD5 hash of the input."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("md5", [self])

    def sha256(self) -> 'FunctionExpression':
        """Return SHA-256 hash of the input."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("sha256", [self])

    def hash(self) -> 'FunctionExpression':
        """Return Murmurhash64 hash of the input."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("hash", [self])

    # ============================================================================
    # UUID FUNCTIONS - MISSING FUNCTIONALITY IMPLEMENTATION
    # ============================================================================

    def uuid(self) -> 'FunctionExpression':
        """Create UUID object from string."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("UUID", [self])

    # ============================================================================
    # BLOB FUNCTIONS - MISSING FUNCTIONALITY IMPLEMENTATION
    # ============================================================================

    def blob(self) -> 'FunctionExpression':
        """Create BLOB object from string."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("BLOB", [self])

    def encode(self) -> 'FunctionExpression':
        """Convert string to blob."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("encode", [self])

    def decode(self) -> 'FunctionExpression':
        """Convert blob to string."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("decode", [self])

    def octet_length(self) -> 'FunctionExpression':
        """Return number of bytes in blob."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("octet_length", [self])

    # ============================================================================
    # STRUCT FUNCTIONS - MISSING FUNCTIONALITY IMPLEMENTATION
    # ============================================================================

    def struct_extract(self, field_name: str) -> 'FunctionExpression':
        """Extract named field from struct."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("struct_extract", [self, field_name])

    # ============================================================================
    # MAP FUNCTIONS - MISSING FUNCTIONALITY IMPLEMENTATION
    # ============================================================================

    def map_extract(self, key: Any) -> 'FunctionExpression':
        """Extract value for given key from map."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("map_extract", [self, key])

    def element_at(self, key: Any) -> 'FunctionExpression':
        """Extract value for given key from map (alias for map_extract)."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("element_at", [self, key])

    def cardinality(self) -> 'FunctionExpression':
        """Return size of the map."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("cardinality", [self])

    def map_keys(self) -> 'FunctionExpression':
        """Return all keys in the map."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("map_keys", [self])

    def map_values(self) -> 'FunctionExpression':
        """Return all values in the map."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("map_values", [self])

    # ============================================================================
    # UNION FUNCTIONS - MISSING FUNCTIONALITY IMPLEMENTATION
    # ============================================================================

    def union_tag(self) -> 'FunctionExpression':
        """Return the tag of union."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("union_tag", [self])

    def union_extract(self, tag: str) -> 'FunctionExpression':
        """Return the value for given tag from union."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("union_extract", [self, tag])

    # ============================================================================
    # NODE/REL FUNCTIONS - MISSING FUNCTIONALITY IMPLEMENTATION
    # ============================================================================

    def id(self) -> 'FunctionExpression':
        """Return internal ID of node/relationship."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("ID", [self])

    def label(self) -> 'FunctionExpression':
        """Return label name of node/relationship."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("LABEL", [self])

    def labels(self) -> 'FunctionExpression':
        """Return label name of node/relationship (alias for label)."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("LABELS", [self])

    def offset(self) -> 'FunctionExpression':
        """Return offset of the internal ID."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("OFFSET", [self])

    # ============================================================================
    # RECURSIVE REL FUNCTIONS - MISSING FUNCTIONALITY IMPLEMENTATION
    # ============================================================================

    def nodes(self) -> 'FunctionExpression':
        """Return all nodes from a path."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("NODES", [self])

    def rels(self) -> 'FunctionExpression':
        """Return all relationships from a path."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("RELS", [self])

    def properties(self, property_name: str) -> 'FunctionExpression':
        """Return given property from nodes or relationships."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("PROPERTIES", [self, property_name])

    def is_trail(self) -> 'FunctionExpression':
        """Check if path contains repeated relationships."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("IS_TRAIL", [self])

    def is_acyclic(self) -> 'FunctionExpression':
        """Check if path contains repeated nodes."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("IS_ACYCLIC", [self])

    def length(self) -> 'FunctionExpression':
        """Return number of relationships (path length) in a path."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("LENGTH", [self])

    def cost(self) -> 'FunctionExpression':
        """Return cost of a weighted path."""
        from .kuzu_query_expressions import FunctionExpression
        return FunctionExpression("COST", [self])

    # ============================================================================
    # CAST AND CASE EXPRESSIONS
    # ============================================================================

    def cast(self, target_type: str, use_as_syntax: bool = False) -> 'CastExpression':
        """Cast this field to target type."""
        from .kuzu_query_expressions import CastExpression
        return CastExpression(self, target_type, use_as_syntax)

    def cast_as(self, target_type: str) -> 'CastExpression':
        """Cast this field to target type using CAST AS syntax."""
        from .kuzu_query_expressions import CastExpression
        return CastExpression(self, target_type, use_as_syntax=True)

    def case(self) -> 'CaseExpression':
        """Create CASE expression with this field as input."""
        from .kuzu_query_expressions import CaseExpression
        return CaseExpression(self)

    # ============================================================================
    # NUMERIC OPERATORS - MISSING FUNCTIONALITY IMPLEMENTATION
    # ============================================================================

    def __add__(self, other: Any) -> 'ArithmeticExpression':
        """Addition operator (+) - handles both numeric addition and list concatenation."""
        from .kuzu_query_expressions import ArithmeticExpression, ArithmeticOperator
        return ArithmeticExpression(self, ArithmeticOperator.ADD, other)

    def __radd__(self, other: Any) -> 'ArithmeticExpression':
        """Right addition operator (+)."""
        from .kuzu_query_expressions import ArithmeticExpression, ArithmeticOperator
        return ArithmeticExpression(other, ArithmeticOperator.ADD, self)

    def __sub__(self, other: Any) -> 'ArithmeticExpression':
        """Subtraction operator (-)."""
        from .kuzu_query_expressions import ArithmeticExpression, ArithmeticOperator
        return ArithmeticExpression(self, ArithmeticOperator.SUB, other)

    def __rsub__(self, other: Any) -> 'ArithmeticExpression':
        """Right subtraction operator (-)."""
        from .kuzu_query_expressions import ArithmeticExpression, ArithmeticOperator
        return ArithmeticExpression(other, ArithmeticOperator.SUB, self)

    def __mul__(self, other: Any) -> 'ArithmeticExpression':
        """Multiplication operator (*)."""
        from .kuzu_query_expressions import ArithmeticExpression, ArithmeticOperator
        return ArithmeticExpression(self, ArithmeticOperator.MUL, other)

    def __rmul__(self, other: Any) -> 'ArithmeticExpression':
        """Right multiplication operator (*)."""
        from .kuzu_query_expressions import ArithmeticExpression, ArithmeticOperator
        return ArithmeticExpression(other, ArithmeticOperator.MUL, self)

    def __truediv__(self, other: Any) -> 'ArithmeticExpression':
        """Division operator (/)."""
        from .kuzu_query_expressions import ArithmeticExpression, ArithmeticOperator
        return ArithmeticExpression(self, ArithmeticOperator.DIV, other)

    def __rtruediv__(self, other: Any) -> 'ArithmeticExpression':
        """Right division operator (/)."""
        from .kuzu_query_expressions import ArithmeticExpression, ArithmeticOperator
        return ArithmeticExpression(other, ArithmeticOperator.DIV, self)

    def __mod__(self, other: Any) -> 'ArithmeticExpression':
        """Modulo operator (%)."""
        from .kuzu_query_expressions import ArithmeticExpression, ArithmeticOperator
        return ArithmeticExpression(self, ArithmeticOperator.MOD, other)

    def __rmod__(self, other: Any) -> 'ArithmeticExpression':
        """Right modulo operator (%)."""
        from .kuzu_query_expressions import ArithmeticExpression, ArithmeticOperator
        return ArithmeticExpression(other, ArithmeticOperator.MOD, self)

    def __pow__(self, other: Any) -> 'ArithmeticExpression':
        """Power operator (^)."""
        from .kuzu_query_expressions import ArithmeticExpression, ArithmeticOperator
        return ArithmeticExpression(self, ArithmeticOperator.POW, other)

    def __rpow__(self, other: Any) -> 'ArithmeticExpression':
        """Right power operator (^)."""
        from .kuzu_query_expressions import ArithmeticExpression, ArithmeticOperator
        return ArithmeticExpression(other, ArithmeticOperator.POW, self)

    # ============================================================================
    # STRING SLICING AND INDEXING OPERATORS
    # ============================================================================

    def __getitem__(self, key: Any) -> 'FunctionExpression':
        """String/array indexing and slicing operator []."""
        from .kuzu_query_expressions import FunctionExpression

        if isinstance(key, slice):
            # Handle slicing [start:end]
            start = key.start if key.start is not None else 1  # Kuzu uses 1-based indexing
            stop = key.stop if key.stop is not None else -1
            return FunctionExpression("array_slice", [self, start, stop])
        else:
            # Handle indexing [index]
            return FunctionExpression("array_extract", [self, key])

    # ============================================================================
    # PATTERN MATCHING OPERATORS
    # ============================================================================

    def regex_match(self, pattern: str) -> FilterExpression:
        """Regex match operator (=~)."""
        return FieldFilterExpression(self.field_path, ComparisonOperator.REGEX_MATCH, pattern)

    def not_regex_match(self, pattern: str) -> FilterExpression:
        """Negative regex match operator (!~)."""
        return FieldFilterExpression(self.field_path, ComparisonOperator.NOT_REGEX_MATCH, pattern)

    # ============================================================================
    # PATTERN MATCHING OPERATOR OVERLOADING
    # ============================================================================

    def __matmul__(self, pattern: str) -> FilterExpression:
        """Pattern matching operator (=~) using @ symbol as Python doesn't support =~."""
        return FieldFilterExpression(self.field_path, ComparisonOperator.REGEX_MATCH, pattern)

    def __rmatmul__(self, pattern: str) -> FilterExpression:
        """Right pattern matching operator (=~) using @ symbol."""
        return FieldFilterExpression(self.field_path, ComparisonOperator.REGEX_MATCH, pattern)

    # ============================================================================
    # TEMPORAL OPERATORS - DATE/TIME/INTERVAL ARITHMETIC
    # ============================================================================

    def date_add(self, value: Any) -> 'TemporalExpression':
        """Add to date field (DATE + INT64 or DATE + INTERVAL)."""
        from .kuzu_query_expressions import TemporalExpression, TemporalOperator

        if isinstance(value, int):
            return TemporalExpression(self, TemporalOperator.DATE_ADD_INT, value)
        else:
            return TemporalExpression(self, TemporalOperator.DATE_ADD_INTERVAL, value)

    def date_sub(self, value: Any) -> 'TemporalExpression':
        """Subtract from date field (DATE - DATE or DATE - INTERVAL)."""
        from .kuzu_query_expressions import TemporalExpression, TemporalOperator

        if isinstance(value, QueryField):
            # It's another field (assume date)
            return TemporalExpression(self, TemporalOperator.DATE_SUB_DATE, value)
        elif isinstance(value, str) and 'INTERVAL' in value.upper():
            # It's an interval string
            return TemporalExpression(self, TemporalOperator.DATE_SUB_INTERVAL, value)
        elif isinstance(value, str):
            # It's a date string
            return TemporalExpression(self, TemporalOperator.DATE_SUB_DATE, value)
        else:
            return TemporalExpression(self, TemporalOperator.DATE_SUB_INTERVAL, value)

    def timestamp_add(self, interval: Any) -> 'TemporalExpression':
        """Add interval to timestamp field (TIMESTAMP + INTERVAL)."""
        from .kuzu_query_expressions import TemporalExpression, TemporalOperator
        return TemporalExpression(self, TemporalOperator.TIMESTAMP_ADD_INTERVAL, interval)

    def timestamp_sub(self, value: Any) -> 'TemporalExpression':
        """Subtract from timestamp field (TIMESTAMP - TIMESTAMP or TIMESTAMP - INTERVAL)."""
        from .kuzu_query_expressions import TemporalExpression, TemporalOperator

        if isinstance(value, QueryField):
            # It's another field (assume timestamp)
            return TemporalExpression(self, TemporalOperator.TIMESTAMP_SUB_TIMESTAMP, value)
        elif isinstance(value, str) and 'INTERVAL' in value.upper():
            # It's an interval string
            return TemporalExpression(self, TemporalOperator.TIMESTAMP_SUB_INTERVAL, value)
        elif isinstance(value, str):
            # It's a timestamp string
            return TemporalExpression(self, TemporalOperator.TIMESTAMP_SUB_TIMESTAMP, value)
        else:
            return TemporalExpression(self, TemporalOperator.TIMESTAMP_SUB_INTERVAL, value)

    def interval_add(self, value: Any) -> 'TemporalExpression':
        """Add to interval field (INTERVAL + INTERVAL/DATE/TIMESTAMP)."""
        from .kuzu_query_expressions import TemporalExpression, TemporalOperator

        # Determine the type of the right operand
        if isinstance(value, str):
            if 'DATE(' in value.upper() and 'TIMESTAMP(' not in value.upper():
                return TemporalExpression(self, TemporalOperator.INTERVAL_ADD_DATE, value)
            elif 'TIMESTAMP(' in value.upper():
                return TemporalExpression(self, TemporalOperator.INTERVAL_ADD_TIMESTAMP, value)
            elif 'INTERVAL(' in value.upper():
                return TemporalExpression(self, TemporalOperator.INTERVAL_ADD_INTERVAL, value)
            else:
                # Default to interval if unclear
                return TemporalExpression(self, TemporalOperator.INTERVAL_ADD_INTERVAL, value)
        else:
            return TemporalExpression(self, TemporalOperator.INTERVAL_ADD_INTERVAL, value)

    def interval_sub(self, interval: Any) -> 'TemporalExpression':
        """Subtract interval from interval field (INTERVAL - INTERVAL)."""
        from .kuzu_query_expressions import TemporalExpression, TemporalOperator
        return TemporalExpression(self, TemporalOperator.INTERVAL_SUB_INTERVAL, interval)


class ModelFieldAccessor:
    """Provides field access for model classes with QueryField descriptors."""
    
    def __init__(self, model_class: Type[Any]):
        self.model_class = model_class
        self._fields_cache: dict[str, QueryField] = {}
    
    def __getattr__(self, name: str) -> QueryField:
        """Get or create QueryField for attribute access."""
        if name.startswith(QueryFieldConstants.PRIVATE_FIELD_PREFIX):
            raise AttributeError(ValidationMessageConstants.CANNOT_ACCESS_PRIVATE_FIELD.format(name))
        
        if name not in self._fields_cache:
            self._fields_cache[name] = QueryField(name, self.model_class)
        
        return self._fields_cache[name]
    
    def get_field(self, name: str) -> QueryField:
        """Explicitly get a field by name."""
        return self.__getattr__(name)

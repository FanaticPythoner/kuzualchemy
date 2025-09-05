# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
Standalone Kuzu functions that don't operate on QueryField objects.
"""

from __future__ import annotations

from typing import Any, List, Optional

from .kuzu_query_expressions import CaseExpression, CastExpression, FunctionExpression


# ============================================================================
# TEXT FUNCTIONS
# ============================================================================

def concat(*args: Any) -> FunctionExpression:
    """Concatenate multiple strings."""
    return FunctionExpression("concat", list(args))


def ws_concat(separator: str, *args: Any) -> FunctionExpression:
    """Concatenate strings with separator."""
    return FunctionExpression("ws_concat", [separator] + list(args))


def array_extract(string_or_list: Any, index: int) -> FunctionExpression:
    """Extract element at index from string or list (1-based)."""
    return FunctionExpression("array_extract", [string_or_list, index])


def array_slice(string_or_list: Any, begin: int, end: int) -> FunctionExpression:
    """Slice string or list from begin to end (1-based)."""
    return FunctionExpression("array_slice", [string_or_list, begin, end])


def list_element(list_value: Any, index: int) -> FunctionExpression:
    """Extract element at index from list (alias for array_extract)."""
    return FunctionExpression("list_element", [list_value, index])


def list_extract(list_value: Any, index: int) -> FunctionExpression:
    """Extract element at index from list (alias for array_extract)."""
    return FunctionExpression("list_extract", [list_value, index])


# ============================================================================
# MISSING TEXT FUNCTIONS FROM DOCUMENTATION
# ============================================================================

def contains(string1: Any, string2: Any) -> FunctionExpression:
    """Returns true if string2 is a substring of string1."""
    return FunctionExpression("contains", [string1, string2])


def ends_with(string1: Any, string2: Any) -> FunctionExpression:
    """Alias of suffix - returns whether string1 ends with string2."""
    return FunctionExpression("ends_with", [string1, string2])


def lower(string: Any) -> FunctionExpression:
    """Returns the string in lower case."""
    return FunctionExpression("lower", [string])


def lcase(string: Any) -> FunctionExpression:
    """Alias of lower - returns the string in lower case."""
    return FunctionExpression("lcase", [string])


def left(string: Any, count: int) -> FunctionExpression:
    """Returns the leftmost count number of characters from string."""
    return FunctionExpression("left", [string, count])


def levenshtein(s1: Any, s2: Any) -> FunctionExpression:
    """Returns the minimum number of single-character edits required to transform s1 to s2."""
    return FunctionExpression("levenshtein", [s1, s2])


def lpad(string: Any, count: int, character: str) -> FunctionExpression:
    """Pads the string with the character from the left until it has count characters."""
    return FunctionExpression("lpad", [string, count, character])


def ltrim(string: Any) -> FunctionExpression:
    """Removes any whitespace in the beginning of the string."""
    return FunctionExpression("ltrim", [string])


def prefix(string: Any, search_string: Any) -> FunctionExpression:
    """Returns whether the string starts with search_string."""
    return FunctionExpression("prefix", [string, search_string])


def repeat(string: Any, count: int) -> FunctionExpression:
    """Repeats the string count number of times."""
    return FunctionExpression("repeat", [string, count])


def reverse(string: Any) -> FunctionExpression:
    """Reverses the string."""
    return FunctionExpression("reverse", [string])


def right(string: Any, count: int) -> FunctionExpression:
    """Returns the rightmost count number of characters from string."""
    return FunctionExpression("right", [string, count])


def rpad(string: Any, count: int, character: str) -> FunctionExpression:
    """Pads the string with the character from the right until it has count characters."""
    return FunctionExpression("rpad", [string, count, character])


def rtrim(string: Any) -> FunctionExpression:
    """Removes any whitespace in the end of the string."""
    return FunctionExpression("rtrim", [string])


def starts_with(string1: Any, string2: Any) -> FunctionExpression:
    """Alias of prefix - returns whether string1 starts with string2."""
    return FunctionExpression("starts_with", [string1, string2])


def substring(string: Any, start: int, length: int) -> FunctionExpression:
    """Extracts the string from start position until length number of characters using 1-based index."""
    return FunctionExpression("substring", [string, start, length])


def substr(string: Any, start: int, length: int) -> FunctionExpression:
    """Alias of substring."""
    return FunctionExpression("substr", [string, start, length])


def suffix(string: Any, search_string: Any) -> FunctionExpression:
    """Returns whether the string ends with search_string."""
    return FunctionExpression("suffix", [string, search_string])


def trim(string: Any) -> FunctionExpression:
    """Removes any whitespace in the beginning or end of the string."""
    return FunctionExpression("trim", [string])


def upper(string: Any) -> FunctionExpression:
    """Returns the string in upper case."""
    return FunctionExpression("upper", [string])


def ucase(string: Any) -> FunctionExpression:
    """Alias of upper - returns the string in upper case."""
    return FunctionExpression("ucase", [string])


def initcap(string: Any) -> FunctionExpression:
    """Returns the string with only the first letter in uppercase."""
    return FunctionExpression("initcap", [string])


def string_split(string: Any, separator: str) -> FunctionExpression:
    """Splits the string along the separator."""
    return FunctionExpression("string_split", [string, separator])


def split_part(string: Any, separator: str, index: int) -> FunctionExpression:
    """Splits the string along the separator and returns the data at the (1-based) index."""
    return FunctionExpression("split_part", [string, separator, index])


# ============================================================================
# NUMERIC FUNCTIONS
# ============================================================================

def pi() -> FunctionExpression:
    """Return the value of pi."""
    return FunctionExpression("pi", [])


def abs(value: Any) -> FunctionExpression:
    """Return absolute value."""
    return FunctionExpression("abs", [value])


def ceil(value: Any) -> FunctionExpression:
    """Round up to next integer."""
    return FunctionExpression("ceil", [value])


def ceiling(value: Any) -> FunctionExpression:
    """Round up to next integer (alias for ceil)."""
    return FunctionExpression("ceiling", [value])


def floor(value: Any) -> FunctionExpression:
    """Round down to nearest integer."""
    return FunctionExpression("floor", [value])


def round(value: Any, precision: int = 0) -> FunctionExpression:
    """Round to specified decimal places."""
    return FunctionExpression("round", [value, precision])


def sqrt(value: Any) -> FunctionExpression:
    """Return square root."""
    return FunctionExpression("sqrt", [value])


def pow(base: Any, exponent: Any) -> FunctionExpression:
    """Return base raised to exponent."""
    return FunctionExpression("pow", [base, exponent])


def sin(value: Any) -> FunctionExpression:
    """Return sine."""
    return FunctionExpression("sin", [value])


def cos(value: Any) -> FunctionExpression:
    """Return cosine."""
    return FunctionExpression("cos", [value])


def tan(value: Any) -> FunctionExpression:
    """Return tangent."""
    return FunctionExpression("tan", [value])


def asin(value: Any) -> FunctionExpression:
    """Return arcsine."""
    return FunctionExpression("asin", [value])


def acos(value: Any) -> FunctionExpression:
    """Return arccosine."""
    return FunctionExpression("acos", [value])


def atan(value: Any) -> FunctionExpression:
    """Return arctangent."""
    return FunctionExpression("atan", [value])


def atan2(x: Any, y: Any) -> FunctionExpression:
    """Return arctangent of x, y."""
    return FunctionExpression("atan2", [x, y])


def ln(value: Any) -> FunctionExpression:
    """Return natural logarithm."""
    return FunctionExpression("ln", [value])


def log(value: Any) -> FunctionExpression:
    """Return base-10 logarithm."""
    return FunctionExpression("log", [value])


def log2(value: Any) -> FunctionExpression:
    """Return base-2 logarithm."""
    return FunctionExpression("log2", [value])


def log10(value: Any) -> FunctionExpression:
    """Return base-10 logarithm (alias for log)."""
    return FunctionExpression("log10", [value])


# ============================================================================
# LIST FUNCTIONS
# ============================================================================

def list_creation(*args: Any) -> FunctionExpression:
    """Create a list containing the argument values."""
    return FunctionExpression("list_creation", list(args))


def size(value: Any) -> FunctionExpression:
    """Return size of string or list."""
    return FunctionExpression("size", [value])


def list_concat(list1: Any, list2: Any) -> FunctionExpression:
    """Concatenate two lists."""
    return FunctionExpression("list_concat", [list1, list2])


def range(start: int, stop: int, step: int = 1) -> FunctionExpression:
    """Return list of values from start to stop with step."""
    if step == 1:
        return FunctionExpression("range", [start, stop])
    else:
        return FunctionExpression("range", [start, stop, step])


# ============================================================================
# MISSING LIST FUNCTIONS FROM DOCUMENTATION
# ============================================================================

def list_cat(list1: Any, list2: Any) -> FunctionExpression:
    """Alias of list_concat."""
    return FunctionExpression("list_cat", [list1, list2])


def array_concat(list1: Any, list2: Any) -> FunctionExpression:
    """Alias of list_concat."""
    return FunctionExpression("array_concat", [list1, list2])


def array_cat(list1: Any, list2: Any) -> FunctionExpression:
    """Alias of list_concat."""
    return FunctionExpression("array_cat", [list1, list2])


def list_append(list_value: Any, element: Any) -> FunctionExpression:
    """Appends the element to list."""
    return FunctionExpression("list_append", [list_value, element])


def array_append(list_value: Any, element: Any) -> FunctionExpression:
    """Alias of list_append."""
    return FunctionExpression("array_append", [list_value, element])


def array_push_back(list_value: Any, element: Any) -> FunctionExpression:
    """Alias of list_append."""
    return FunctionExpression("array_push_back", [list_value, element])


def list_prepend(list_value: Any, element: Any) -> FunctionExpression:
    """Prepends the element to list."""
    return FunctionExpression("list_prepend", [list_value, element])


def array_prepend(list_value: Any, element: Any) -> FunctionExpression:
    """Alias of list_prepend."""
    return FunctionExpression("array_prepend", [list_value, element])


def array_push_front(list_value: Any, element: Any) -> FunctionExpression:
    """Alias of list_prepend."""
    return FunctionExpression("array_push_front", [list_value, element])


def list_position(list_value: Any, element: Any) -> FunctionExpression:
    """Returns the position of element in the list."""
    return FunctionExpression("list_position", [list_value, element])


def list_indexof(list_value: Any, element: Any) -> FunctionExpression:
    """Alias of list_position."""
    return FunctionExpression("list_indexof", [list_value, element])


def array_position(list_value: Any, element: Any) -> FunctionExpression:
    """Alias of list_position."""
    return FunctionExpression("array_position", [list_value, element])


def array_indexof(list_value: Any, element: Any) -> FunctionExpression:
    """Alias of list_position."""
    return FunctionExpression("array_indexof", [list_value, element])


def list_contains(list_value: Any, element: Any) -> FunctionExpression:
    """Returns true if the list contains the element."""
    return FunctionExpression("list_contains", [list_value, element])


def list_has(list_value: Any, element: Any) -> FunctionExpression:
    """Alias of list_contains."""
    return FunctionExpression("list_has", [list_value, element])


def array_contains(list_value: Any, element: Any) -> FunctionExpression:
    """Alias of list_contains."""
    return FunctionExpression("array_contains", [list_value, element])


def array_has(list_value: Any, element: Any) -> FunctionExpression:
    """Alias of list_contains."""
    return FunctionExpression("array_has", [list_value, element])


def list_slice(list_value: Any, begin: int, end: int) -> FunctionExpression:
    """Extracts a sub-list using slice conventions."""
    return FunctionExpression("list_slice", [list_value, begin, end])


def list_reverse(list_value: Any) -> FunctionExpression:
    """Reverse list elements."""
    return FunctionExpression("list_reverse", [list_value])


def list_sort(list_value: Any, order: str = "ASC", nulls: str = "NULLS FIRST") -> FunctionExpression:
    """Sorts the elements of the list."""
    if order == "ASC" and nulls == "NULLS FIRST":
        return FunctionExpression("list_sort", [list_value])
    elif order != "ASC" and nulls == "NULLS FIRST":
        return FunctionExpression("list_sort", [list_value, order])
    else:
        return FunctionExpression("list_sort", [list_value, order, nulls])


def list_reverse_sort(list_value: Any) -> FunctionExpression:
    """Alias of list_sort(list, 'DESC')."""
    return FunctionExpression("list_reverse_sort", [list_value])


def list_sum(list_value: Any) -> FunctionExpression:
    """Sums the elements of the list."""
    return FunctionExpression("list_sum", [list_value])


def list_product(list_value: Any) -> FunctionExpression:
    """Multiply elements of the list."""
    return FunctionExpression("list_product", [list_value])


def list_distinct(list_value: Any) -> FunctionExpression:
    """Removes NULLs and duplicate values from the list."""
    return FunctionExpression("list_distinct", [list_value])


def list_unique(list_value: Any) -> FunctionExpression:
    """Counts number of unique elements of the list."""
    return FunctionExpression("list_unique", [list_value])


def list_any_value(list_value: Any) -> FunctionExpression:
    """Returns the first non-NULL value of the list."""
    return FunctionExpression("list_any_value", [list_value])


def list_to_string(separator: str, list_value: Any) -> FunctionExpression:
    """Converts a list to a string separated by the given separator."""
    return FunctionExpression("list_to_string", [separator, list_value])


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def coalesce(*args: Any) -> FunctionExpression:
    """Return first non-NULL value."""
    return FunctionExpression("coalesce", list(args))


def ifnull(value: Any, replacement: Any) -> FunctionExpression:
    """Return replacement if value is NULL."""
    return FunctionExpression("ifnull", [value, replacement])


def nullif(value1: Any, value2: Any) -> FunctionExpression:
    """Return NULL if value1 equals value2, otherwise return value1."""
    return FunctionExpression("nullif", [value1, value2])


def typeof(value: Any) -> FunctionExpression:
    """Return type of value."""
    return FunctionExpression("typeof", [value])


# ============================================================================
# DATE/TIME FUNCTIONS
# ============================================================================

def current_date() -> FunctionExpression:
    """Return current date."""
    return FunctionExpression("current_date", [])


def current_timestamp() -> FunctionExpression:
    """Return current timestamp."""
    return FunctionExpression("current_timestamp", [])


def date_part(part: str, date_value: Any) -> FunctionExpression:
    """Extract part from date/timestamp."""
    return FunctionExpression("date_part", [part, date_value])


def date_trunc(part: str, date_value: Any) -> FunctionExpression:
    """Truncate date/timestamp to specified part."""
    return FunctionExpression("date_trunc", [part, date_value])


def datepart(part: str, date_value: Any) -> FunctionExpression:
    """Extract part from date (alias for date_part)."""
    return FunctionExpression("datepart", [part, date_value])


def datetrunc(part: str, date_value: Any) -> FunctionExpression:
    """Truncate date to specified precision (alias for date_trunc)."""
    return FunctionExpression("datetrunc", [part, date_value])


def dayname(date_value: Any) -> FunctionExpression:
    """Get English name of the day."""
    return FunctionExpression("dayname", [date_value])


def monthname(date_value: Any) -> FunctionExpression:
    """Get English name of the month."""
    return FunctionExpression("monthname", [date_value])


def last_day(date_value: Any) -> FunctionExpression:
    """Get last day of the month."""
    return FunctionExpression("last_day", [date_value])


def greatest(*args: Any) -> FunctionExpression:
    """Return the greatest value."""
    return FunctionExpression("greatest", list(args))


def least(*args: Any) -> FunctionExpression:
    """Return the least value."""
    return FunctionExpression("least", list(args))


def make_date(year: int, month: int, day: int) -> FunctionExpression:
    """Create date from year, month, day."""
    return FunctionExpression("make_date", [year, month, day])


def century(timestamp_value: Any) -> FunctionExpression:
    """Get century from timestamp."""
    return FunctionExpression("century", [timestamp_value])


def epoch_ms(ms_value: Any) -> FunctionExpression:
    """Convert milliseconds to timestamp."""
    return FunctionExpression("epoch_ms", [ms_value])


def to_epoch_ms(timestamp_value: Any) -> FunctionExpression:
    """Convert timestamp to milliseconds."""
    return FunctionExpression("to_epoch_ms", [timestamp_value])


# ============================================================================
# HASH FUNCTIONS
# ============================================================================

def md5(value: Any) -> FunctionExpression:
    """Return MD5 hash."""
    return FunctionExpression("md5", [value])


def sha256(value: Any) -> FunctionExpression:
    """Return SHA256 hash."""
    return FunctionExpression("sha256", [value])


def hash(value: Any) -> FunctionExpression:
    """Return hash value."""
    return FunctionExpression("hash", [value])


# ============================================================================
# UUID FUNCTIONS
# ============================================================================

def gen_random_uuid() -> FunctionExpression:
    """Generate random UUID."""
    return FunctionExpression("gen_random_uuid", [])


def uuid(value: Any) -> FunctionExpression:
    """Create UUID object from string."""
    return FunctionExpression("UUID", [value])


# ============================================================================
# CASTING FUNCTIONS (moved to CAST AND CASE EXPRESSIONS section)
# ============================================================================


def to_int64(value: Any) -> FunctionExpression:
    """Cast to INT64."""
    return FunctionExpression("to_int64", [value])


def to_int32(value: Any) -> FunctionExpression:
    """Cast to INT32."""
    return FunctionExpression("to_int32", [value])


def to_int16(value: Any) -> FunctionExpression:
    """Cast to INT16."""
    return FunctionExpression("to_int16", [value])


def to_double(value: Any) -> FunctionExpression:
    """Cast to DOUBLE."""
    return FunctionExpression("to_double", [value])


def to_float(value: Any) -> FunctionExpression:
    """Cast to FLOAT."""
    return FunctionExpression("to_float", [value])


def to_string(value: Any) -> FunctionExpression:
    """Cast to STRING."""
    return FunctionExpression("to_string", [value])


def to_date(value: Any) -> FunctionExpression:
    """Cast to DATE."""
    return FunctionExpression("to_date", [value])


def to_timestamp(value: Any) -> FunctionExpression:
    """Cast to TIMESTAMP."""
    return FunctionExpression("to_timestamp", [value])


# ============================================================================
# INTERVAL FUNCTIONS
# ============================================================================

def to_years(value: Any) -> FunctionExpression:
    """Convert integer to year interval."""
    return FunctionExpression("to_years", [value])


def to_months(value: Any) -> FunctionExpression:
    """Convert integer to month interval."""
    return FunctionExpression("to_months", [value])


def to_days(value: Any) -> FunctionExpression:
    """Convert integer to day interval."""
    return FunctionExpression("to_days", [value])


def to_hours(value: Any) -> FunctionExpression:
    """Convert integer to hour interval."""
    return FunctionExpression("to_hours", [value])


def to_minutes(value: Any) -> FunctionExpression:
    """Convert integer to minute interval."""
    return FunctionExpression("to_minutes", [value])


def to_seconds(value: Any) -> FunctionExpression:
    """Convert integer to second interval."""
    return FunctionExpression("to_seconds", [value])


def to_milliseconds(value: Any) -> FunctionExpression:
    """Convert integer to millisecond interval."""
    return FunctionExpression("to_milliseconds", [value])


def to_microseconds(value: Any) -> FunctionExpression:
    """Convert integer to microsecond interval."""
    return FunctionExpression("to_microseconds", [value])


# ============================================================================
# ARRAY FUNCTIONS
# ============================================================================

def array_value(*args: Any) -> FunctionExpression:
    """Create array containing the argument values."""
    return FunctionExpression("array_value", list(args))


def array_distance(array1: Any, array2: Any) -> FunctionExpression:
    """Calculate Euclidean distance between two arrays."""
    return FunctionExpression("array_distance", [array1, array2])


def array_squared_distance(array1: Any, array2: Any) -> FunctionExpression:
    """Calculate squared Euclidean distance between two arrays."""
    return FunctionExpression("array_squared_distance", [array1, array2])


def array_dot_product(array1: Any, array2: Any) -> FunctionExpression:
    """Calculate dot product of two arrays."""
    return FunctionExpression("array_dot_product", [array1, array2])


def array_inner_product(array1: Any, array2: Any) -> FunctionExpression:
    """Calculate inner product of two arrays (alias for dot product)."""
    return FunctionExpression("array_inner_product", [array1, array2])


def array_cross_product(array1: Any, array2: Any) -> FunctionExpression:
    """Calculate cross product of two arrays."""
    return FunctionExpression("array_cross_product", [array1, array2])


def array_cosine_similarity(array1: Any, array2: Any) -> FunctionExpression:
    """Calculate cosine similarity of two arrays."""
    return FunctionExpression("array_cosine_similarity", [array1, array2])


# ============================================================================
# ADDITIONAL UTILITY FUNCTIONS
# ============================================================================

def constant_or_null(constant: Any, check_value: Any) -> FunctionExpression:
    """Return constant if check_value is not NULL, otherwise return NULL."""
    return FunctionExpression("constant_or_null", [constant, check_value])


def count_if(condition: Any) -> FunctionExpression:
    """Return 1 if condition is true or non-zero, otherwise 0."""
    return FunctionExpression("count_if", [condition])


def error(message: Any) -> FunctionExpression:
    """Throw message as a runtime exception."""
    return FunctionExpression("error", [message])


# ============================================================================
# BLOB FUNCTIONS
# ============================================================================

def blob(value: Any) -> FunctionExpression:
    """Create BLOB object from string."""
    return FunctionExpression("BLOB", [value])


def encode(value: Any) -> FunctionExpression:
    """Convert string to blob."""
    return FunctionExpression("encode", [value])


def decode(value: Any) -> FunctionExpression:
    """Convert blob to string."""
    return FunctionExpression("decode", [value])


def octet_length(value: Any) -> FunctionExpression:
    """Return number of bytes in blob."""
    return FunctionExpression("octet_length", [value])


# ============================================================================
# STRUCT FUNCTIONS
# ============================================================================

def struct_extract(struct_value: Any, field_name: str) -> FunctionExpression:
    """Extract named field from struct."""
    return FunctionExpression("struct_extract", [struct_value, field_name])


# ============================================================================
# MAP FUNCTIONS
# ============================================================================

def map_func(keys: Any, values: Any) -> FunctionExpression:
    """Create a map from keys and values."""
    return FunctionExpression("map", [keys, values])


def map_extract(map_value: Any, key: Any) -> FunctionExpression:
    """Extract value for given key from map."""
    return FunctionExpression("map_extract", [map_value, key])


def element_at(map_value: Any, key: Any) -> FunctionExpression:
    """Extract value for given key from map (alias for map_extract)."""
    return FunctionExpression("element_at", [map_value, key])


def cardinality(map_value: Any) -> FunctionExpression:
    """Return size of the map."""
    return FunctionExpression("cardinality", [map_value])


def map_keys(map_value: Any) -> FunctionExpression:
    """Return all keys in the map."""
    return FunctionExpression("map_keys", [map_value])


def map_values(map_value: Any) -> FunctionExpression:
    """Return all values in the map."""
    return FunctionExpression("map_values", [map_value])


# ============================================================================
# UNION FUNCTIONS
# ============================================================================

def union_value(tag: str, value: Any) -> FunctionExpression:
    """Create union with given value and tag."""
    return FunctionExpression("union_value", [f"{tag} := {value}"])


def union_tag(union_value: Any) -> FunctionExpression:
    """Return the tag of union."""
    return FunctionExpression("union_tag", [union_value])


def union_extract(union_value: Any, tag: str) -> FunctionExpression:
    """Return the value for given tag from union."""
    return FunctionExpression("union_extract", [union_value, tag])


# ============================================================================
# NODE/REL FUNCTIONS
# ============================================================================

def id_func(node_or_rel: Any) -> FunctionExpression:
    """Return internal ID of node/relationship."""
    return FunctionExpression("ID", [node_or_rel])


def label(node_or_rel: Any) -> FunctionExpression:
    """Return label name of node/relationship."""
    return FunctionExpression("LABEL", [node_or_rel])


def labels(node_or_rel: Any) -> FunctionExpression:
    """Return label name of node/relationship (alias for label)."""
    return FunctionExpression("LABELS", [node_or_rel])


def offset(node_or_rel: Any) -> FunctionExpression:
    """Return offset of the internal ID."""
    return FunctionExpression("OFFSET", [node_or_rel])


# ============================================================================
# RECURSIVE REL FUNCTIONS
# ============================================================================

def nodes(path: Any) -> FunctionExpression:
    """Return all nodes from a path."""
    return FunctionExpression("NODES", [path])


def rels(path: Any) -> FunctionExpression:
    """Return all relationships from a path."""
    return FunctionExpression("RELS", [path])


def properties(path: Any, property_name: str) -> FunctionExpression:
    """Return given property from nodes or relationships."""
    return FunctionExpression("PROPERTIES", [path, property_name])


def is_trail(path: Any) -> FunctionExpression:
    """Check if path contains repeated relationships."""
    return FunctionExpression("IS_TRAIL", [path])


def is_acyclic(path: Any) -> FunctionExpression:
    """Check if path contains repeated nodes."""
    return FunctionExpression("IS_ACYCLIC", [path])


def length(path: Any) -> FunctionExpression:
    """Return number of relationships (path length) in a path."""
    return FunctionExpression("LENGTH", [path])


def cost(path: Any) -> FunctionExpression:
    """Return cost of a weighted path."""
    return FunctionExpression("COST", [path])


# ============================================================================
# ADDITIONAL MISSING NUMERIC FUNCTIONS
# ============================================================================

def negate(value: Any) -> FunctionExpression:
    """Return negative value."""
    return FunctionExpression("negate", [value])


def sign(value: Any) -> FunctionExpression:
    """Return sign (-1, 0, or 1)."""
    return FunctionExpression("sign", [value])


def even(value: Any) -> FunctionExpression:
    """Round to next even number."""
    return FunctionExpression("even", [value])


def factorial(value: Any) -> FunctionExpression:
    """Return factorial."""
    return FunctionExpression("factorial", [value])


def gamma(value: Any) -> FunctionExpression:
    """Return gamma function."""
    return FunctionExpression("gamma", [value])


def lgamma(value: Any) -> FunctionExpression:
    """Return log of gamma function."""
    return FunctionExpression("lgamma", [value])


def bitwise_xor(x: Any, y: Any) -> FunctionExpression:
    """Return bitwise XOR."""
    return FunctionExpression("bitwise_xor", [x, y])


def cot(value: Any) -> FunctionExpression:
    """Return cotangent."""
    return FunctionExpression("cot", [value])


def degrees(value: Any) -> FunctionExpression:
    """Convert radians to degrees."""
    return FunctionExpression("degrees", [value])


def radians(value: Any) -> FunctionExpression:
    """Convert degrees to radians."""
    return FunctionExpression("radians", [value])


# ============================================================================
# CAST AND CASE EXPRESSIONS
# ============================================================================

def cast(value: Any, target_type: str, use_as_syntax: bool = False) -> CastExpression:
    """Cast value to target type using CAST function."""
    from .kuzu_query_expressions import CastExpression
    return CastExpression(value, target_type, use_as_syntax)


def cast_as(value: Any, target_type: str) -> CastExpression:
    """Cast value to target type using CAST AS syntax."""
    from .kuzu_query_expressions import CastExpression
    return CastExpression(value, target_type, use_as_syntax=True)


def case(input_expr: Any = None) -> CaseExpression:
    """Create CASE expression for conditional logic."""
    from .kuzu_query_expressions import CaseExpression
    return CaseExpression(input_expr)


# ============================================================================
# ADDITIONAL LIST FUNCTIONS (DOCUMENTED IN KUZU)
# ============================================================================

def list_transform(list_value: Any, lambda_expr: str) -> FunctionExpression:
    """Transform list elements using lambda expression."""
    return FunctionExpression("list_transform", [list_value, lambda_expr])


def list_filter(list_value: Any, lambda_expr: str) -> FunctionExpression:
    """Filter list elements using lambda expression."""
    return FunctionExpression("list_filter", [list_value, lambda_expr])


def list_reduce(list_value: Any, lambda_expr: str) -> FunctionExpression:
    """Reduce list to single value using lambda expression."""
    return FunctionExpression("list_reduce", [list_value, lambda_expr])


def list_has_all(list_value: Any, sub_list: Any) -> FunctionExpression:
    """Check if list contains all elements from sub-list."""
    return FunctionExpression("list_has_all", [list_value, sub_list])


def all_func(variable: str, list_value: Any, predicate: str) -> FunctionExpression:
    """Check if all elements in list satisfy predicate."""
    return FunctionExpression("all", [f"{variable} IN {list_value} WHERE {predicate}"])


def any_func(variable: str, list_value: Any, predicate: str) -> FunctionExpression:
    """Check if any element in list satisfies predicate."""
    return FunctionExpression("any", [f"{variable} IN {list_value} WHERE {predicate}"])


def none_func(variable: str, list_value: Any, predicate: str) -> FunctionExpression:
    """Check if no element in list satisfies predicate."""
    return FunctionExpression("none", [f"{variable} IN {list_value} WHERE {predicate}"])


def single_func(variable: str, list_value: Any, predicate: str) -> FunctionExpression:
    """Check if exactly one element in list satisfies predicate."""
    return FunctionExpression("single", [f"{variable} IN {list_value} WHERE {predicate}"])


# ============================================================================
# PATTERN MATCHING FUNCTIONS (DOCUMENTED IN KUZU)
# ============================================================================

def regexp_matches(string: Any, pattern: str) -> FunctionExpression:
    """Return true if a part of string matches the regex."""
    return FunctionExpression("regexp_matches", [string, pattern])


def regexp_replace(string: Any, pattern: str, replacement: str, options: str | None = None) -> FunctionExpression:
    """Replace the matching part of string with replacement."""
    if options is not None:
        return FunctionExpression("regexp_replace", [string, pattern, replacement, options])
    else:
        return FunctionExpression("regexp_replace", [string, pattern, replacement])


def regexp_extract(string: Any, pattern: str, group: int = 0) -> FunctionExpression:
    """Split the string along the regex and extract the first occurrence of the group."""
    return FunctionExpression("regexp_extract", [string, pattern, group])


def regexp_extract_all(string: Any, pattern: str, group: int = 0) -> FunctionExpression:
    """Split the string along the regex and extract all occurrences of the group."""
    return FunctionExpression("regexp_extract_all", [string, pattern, group])


def regexp_split_to_array(string: Any, pattern: str, options: str | None = None) -> FunctionExpression:
    """Split the string along the regex and extract all occurrences between regex."""
    if options is not None:
        return FunctionExpression("regexp_split_to_array", [string, pattern, options])
    else:
        return FunctionExpression("regexp_split_to_array", [string, pattern])



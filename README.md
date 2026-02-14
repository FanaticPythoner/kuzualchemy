# KuzuAlchemy

A SQLAlchemy-like ORM for Kuzu graph database

[![Tests](https://github.com/FanaticPythoner/kuzualchemy/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/FanaticPythoner/kuzualchemy/actions/workflows/ci.yml?query=branch%3Amain)
[![PyPI version](https://img.shields.io/pypi/v/kuzualchemy.svg)](https://pypi.org/project/kuzualchemy/)
[![Python versions](https://img.shields.io/pypi/pyversions/kuzualchemy.svg)](https://pypi.org/project/kuzualchemy/)

KuzuAlchemy is an Object-Relational Mapping (ORM) library for the [Kuzu graph database](https://kuzudb.com/). It provides a SQLAlchemy-like interface for working with graph data.

> **Note**: This software is currently in alpha development. APIs may change.


## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Function Reference](#function-reference)
5. [Operator Reference](#operator-reference)
6. [Model Definition](#model-definition)
7. [Field Types & Metadata](#field-types--metadata)
8. [Relationships](#relationships)
9. [Query System](#query-system)
10. [Session Management](#session-management)
11. [Advanced Features](#advanced-features)
12. [API Reference](#api-reference)
13. [Contributing](#contributing)
14. [License](#license)

## Overview

KuzuAlchemy provides the following components:

- **Core ORM** (`kuzu_orm.py`): Base classes for nodes and relationships with metadata handling
- **Session Management** (`kuzu_session.py`): Database operations with transaction support
- **Query System** (`kuzu_query.py`): Query builder with Cypher generation
- **Expression Engine** (`kuzu_query_expressions.py`): Expression system supporting Kuzu operators
- **Function Library** (`kuzu_functions.py`): Kuzu functions implemented as standalone callables
- **Field Integration** (`kuzu_query_fields.py`): QueryField methods providing fluent API access to functions

### Key Features

- **Kuzu Function Support**: Kuzu functions and operators implemented
- **ORM**: Model definition, session management, and querying capabilities
- **Type-Safe Operations**: Type safety with parameter handling and validation
- **Testing**: Test coverage for functionality
- **Error Handling**: Error handling and transaction management

## Installation

### Prerequisites

```bash
pip install kuzu pydantic
```

### Install KuzuAlchemy

```bash
pip install kuzualchemy
```

### Development Installation

```bash
git clone <repository-url>
cd kuzualchemy
pip install -e ".[dev,test]"
```

### Internal distribution workflow (CodeGraph)

Use this workflow for internal builds/releases with CGPM + devpi.

```bash
# From /home/n00ne/Documents/GitHub/cgpm
./.venv/bin/cgpm -c config/packages.toml pull kuzualchemy
./.venv/bin/cgpm -c config/packages.toml build kuzualchemy --python ./.venv/bin/python
./.venv/bin/cgpm -c config/packages.toml publish kuzualchemy --python ./.venv/bin/python
```

Internal consumer install:

```bash
python -m pip install --extra-index-url http://localhost:3141/codegraph/prod/+simple/ kuzualchemy
```

Internal dependency declaration policy (`pyproject.toml`):

```toml
[project]
dependencies = [
    "pydantic>=2.0.0",
    "atp-pipeline",
]
```

`atp-pipeline` is an internal package and is intentionally unpinned (bare name only).

## Quick Start

### Basic Setup

```python
from kuzualchemy import (
    KuzuBaseModel, KuzuRelationshipBase,
    kuzu_node, kuzu_relationship, kuzu_field,
    KuzuDataType, KuzuSession,
    get_all_ddl
)

# Create session
session = KuzuSession(db_path="database.db")

# Initialize schema
ddl = get_all_ddl()
if ddl.strip():
    session.execute(ddl)
```

### Example

```python
import kuzualchemy as ka
from pathlib import Path

# Define your graph models
@ka.kuzu_node("Person")
class Person(ka.KuzuBaseModel):
    name: str = ka.kuzu_field(kuzu_type=ka.KuzuDataType.STRING, primary_key=True)
    age: int = ka.kuzu_field(kuzu_type=ka.KuzuDataType.INT32)
    email: str = ka.kuzu_field(kuzu_type=ka.KuzuDataType.STRING)

@ka.kuzu_relationship("KNOWS", pairs=[(Person, Person)])
class Knows(ka.KuzuRelationshipBase):
    since: int = ka.kuzu_field(kuzu_type=ka.KuzuDataType.INT32)
    strength: float = ka.kuzu_field(kuzu_type=ka.KuzuDataType.DOUBLE, default=1.0)

# Create database and session
db_path = Path("my_graph.db")
session = ka.KuzuSession(db_path)

# Create schema
session.execute(ka.get_all_ddl())

# Insert data
alice = Person(name="Alice", age=30, email="alice@example.com")
bob = Person(name="Bob", age=25, email="bob@example.com")
knows = Knows(from_node=alice, to_node=bob, since=2020, strength=0.9)

# Or, you could do `session.add_all([alice, bob, knows])`
session.add(alice)
session.add(bob)
session.add(knows)
session.commit()

# Query data
query = ka.Query(Person, session=session)
filtered_query = query.where(query.fields.age > 25)
results = filtered_query.all()

print(f"Found {len(results)} people over 25")
```

---

## Function Reference

KuzuAlchemy implements Kuzu functions across multiple categories. Each function returns a `FunctionExpression` object that can be used in queries and expressions.

### Text Functions

String manipulation and text processing functions:

| Function | Description | kuzu_functions | QueryField |
|----------|-------------|----------------|------------|
| `concat(*args)` | Concatenate multiple strings | ✓ | ✓ |
| `ws_concat(separator, *args)` | Concatenate strings with separator | ✓ | — |
| `array_extract(string_or_list, index)` | Extract element at 1-based index from string or list | ✓ | — |
| `array_slice(string_or_list, begin, end)` | Slice string or list (1-based) | ✓ | — |
| `list_element(list_value, index)` | Extract list element at index | ✓ | ✓ |
| `list_extract(list_value, index)` | Extract list element at index (alias) | ✓ | ✓ |
| `contains(string1, string2)` | Substring test | ✓ | ✓ |
| `ends_with(string1, string2)` | Ends-with test (alias of suffix) | ✓ | ✓ |
| `lower(string)` | Lowercase | ✓ | ✓ |
| `lcase(string)` | Lowercase (alias) | ✓ | ✓ |
| `left(string, count)` | Left substring | ✓ | ✓ |
| `levenshtein(s1, s2)` | Edit distance | ✓ | ✓ |
| `lpad(string, count, character)` | Left pad | ✓ | ✓ |
| `ltrim(string)` | Trim left | ✓ | ✓ |
| `prefix(string, search_string)` | Starts-with test | ✓ | — |
| `repeat(string, count)` | Repeat string | ✓ | ✓ |
| `reverse(string)` | Reverse string | ✓ | ✓ |
| `right(string, count)` | Right substring | ✓ | ✓ |
| `rpad(string, count, character)` | Right pad | ✓ | ✓ |
| `rtrim(string)` | Trim right | ✓ | ✓ |
| `starts_with(string1, string2)` | Starts-with test (alias of prefix) | ✓ | ✓ |
| `substring(string, start, length)` | Substring by 1-based start/length | ✓ | ✓ |
| `substr(string, start, length)` | Substring (alias) | ✓ | ✓ |
| `suffix(string, search_string)` | Ends-with test | ✓ | — |
| `trim(string)` | Trim both sides | ✓ | ✓ |
| `upper(string)` | Uppercase | ✓ | ✓ |
| `ucase(string)` | Uppercase (alias) | ✓ | ✓ |
| `initcap(string)` | Capitalize first letter | ✓ | ✓ |
| `string_split(string, separator)` | Split to array | ✓ | ✓ |
| `split_part(string, separator, index)` | Part at 1-based index | ✓ | ✓ |

### Pattern Matching Functions

Regular expression utilities:

| Function | Description | kuzu_functions | QueryField |
|----------|-------------|----------------|------------|
| `regexp_matches(string, pattern)` | Regex test | ✓ | ✓ |
| `regexp_replace(string, pattern, replacement[, options])` | Regex replace | ✓ | ✓ |
| `regexp_extract(string, pattern[, group])` | Extract first match/group | ✓ | ✓ |
| `regexp_extract_all(string, pattern[, group])` | Extract all matches/groups | ✓ | ✓ |
| `regexp_split_to_array(string, pattern[, options])` | Split by regex | ✓ | ✓ |


### List Functions

Array and list manipulation functions:

| Function | Description | kuzu_functions | QueryField |
|----------|-------------|----------------|------------|
| `list_creation(...)` | Create a list containing the argument values | ✓ | — |
| `size(value)` | Return size of string or list | ✓ | ✓ |
| `list_concat(list1, list2)` | Concatenate two lists | ✓ | ✓ |
| `range(start, stop, [step])` | Return list from start to stop with step | ✓ | — |
| `list_cat(list1, list2)` | Alias of list_concat | ✓ | ✓ |
| `array_concat(list1, list2)` | Alias of list_concat | ✓ | ✓ |
| `array_cat(list1, list2)` | Alias of list_concat | ✓ | ✓ |
| `list_append(list, element)` | Append element to list | ✓ | ✓ |
| `array_append(list, element)` | Alias of list_append | ✓ | ✓ |
| `array_push_back(list, element)` | Alias of list_append | ✓ | ✓ |
| `list_prepend(list, element)` | Prepend element to list | ✓ | ✓ |
| `array_prepend(list, element)` | Alias of list_prepend | ✓ | ✓ |
| `array_push_front(list, element)` | Alias of list_prepend | ✓ | ✓ |
| `list_position(list, element)` | Position of element in list | ✓ | ✓ |
| `list_indexof(list, element)` | Alias of list_position | ✓ | ✓ |
| `array_position(list, element)` | Alias of list_position | ✓ | ✓ |
| `array_indexof(list, element)` | Alias of list_position | ✓ | ✓ |
| `list_contains(list, element)` | Check if list contains element | ✓ | ✓ |
| `list_has(list, element)` | Alias of list_contains | ✓ | ✓ |
| `array_contains(list, element)` | Alias of list_contains | ✓ | ✓ |
| `array_has(list, element)` | Alias of list_contains | ✓ | ✓ |
| `list_slice(list, begin, end)` | Extract sub-list | ✓ | ✓ |

### Advanced List Functions

Higher-order and quantifier list functions (order matches kuzu_functions.py):

| Function | Description | kuzu_functions | QueryField |
|----------|-------------|----------------|------------|
| `list_reverse(list)` | Reverse list elements | ✓ | ✓ |
| `list_sort(list[, order, nulls])` | Sort elements of list | ✓ | ✓ |
| `list_reverse_sort(list)` | Sort elements of list in DESC | ✓ | ✓ |
| `list_sum(list)` | Sum elements | ✓ | ✓ |
| `list_product(list)` | Multiply elements | ✓ | ✓ |
| `list_distinct(list)` | Remove NULLs and duplicates | ✓ | ✓ |
| `list_unique(list)` | Count unique elements | ✓ | ✓ |
| `list_any_value(list)` | First non-NULL value | ✓ | ✓ |
| `list_to_string(sep, list)` | Join elements with separator | ✓ | ✓ |
| `list_transform(list, lambda)` | Transform elements using lambda expression | ✓ | ✓ |
| `list_filter(list, lambda)` | Filter elements using lambda expression | ✓ | ✓ |
| `list_reduce(list, lambda)` | Reduce list using lambda expression | ✓ | ✓ |
| `list_has_all(list, sub_list)` | Contains all elements from sub-list | ✓ | ✓ |
| `all_func(var, list, predicate)` | All elements satisfy predicate | ✓ | ✓ |
| `any_func(var, list, predicate)` | Any element satisfies predicate | ✓ | ✓ |
| `none_func(var, list, predicate)` | No elements satisfy predicate | ✓ | ✓ |
| `single_func(var, list, predicate)` | Exactly one element satisfies predicate | ✓ | ✓ |

| `array_slice(array, start, end)` | Slice array | `ka.array_slice(field, 1, 5)` |
| `list_reverse(list)` | Reverse list | `ka.list_reverse(field)` |
| `list_sort(list)` | Sort list | `ka.list_sort(field)` |
| `list_reverse_sort(list)` | Reverse sort | `ka.list_reverse_sort(field)` |
| `list_sum(list)` | Sum elements | `ka.list_sum(field)` |
| `list_product(list)` | Product elements | `ka.list_product(field)` |
| `list_distinct(list)` | Distinct elements | `ka.list_distinct(field)` |
| `list_unique(list)` | Unique elements | `ka.list_unique(field)` |
| `list_any_value(list)` | Any element | `ka.list_any_value(field)` |
| `list_to_string(sep, list)` | Join elements | `ka.list_to_string(",", field)` |
| `list_extract(list, index)` | Extract element | `ka.list_extract(field, 1)` |
| `list_element(list, index)` | Get element | `ka.list_element(field, 1)` |
| `range(start, end, [step])` | Generate range | `ka.range(1, 10)` |

### Numeric Functions

Mathematical and numeric computation functions:

| Function | Description | kuzu_functions | QueryField |
|----------|-------------|----------------|------------|
| `pi()` | Return value of pi | ✓ | — |
| `abs(value)` | Absolute value | ✓ | ✓ |
| `ceil(value)` | Ceiling | ✓ | ✓ |
| `ceiling(value)` | Ceiling (alias) | ✓ | ✓ |
| `floor(value)` | Floor | ✓ | ✓ |
| `round(value, precision=0)` | Round to precision | ✓ | ✓ |
| `sqrt(value)` | Square root | ✓ | ✓ |
| `pow(base, exponent)` | Power | ✓ | ✓ |
| `sin(value)` | Sine | ✓ | ✓ |
| `cos(value)` | Cosine | ✓ | ✓ |
| `tan(value)` | Tangent | ✓ | ✓ |
| `asin(value)` | Arcsine | ✓ | ✓ |
| `acos(value)` | Arccosine | ✓ | ✓ |
| `atan(value)` | Arctangent | ✓ | ✓ |
| `atan2(x, y)` | Arctangent of x,y | ✓ | ✓ |
| `ln(value)` | Natural log | ✓ | ✓ |
| `log(value)` | Logarithm | ✓ | ✓ |
| `log2(value)` | Base-2 logarithm | ✓ | ✓ |
| `log10(value)` | Base-10 logarithm | ✓ | ✓ |
| `negate(value)` | Negation | ✓ | ✓ |
| `sign(value)` | Sign (-1,0,1) | ✓ | ✓ |
| `even(value)` | Round to next even | ✓ | ✓ |
| `factorial(value)` | Factorial | ✓ | ✓ |
| `gamma(value)` | Gamma function | ✓ | ✓ |
| `lgamma(value)` | Log Gamma | ✓ | ✓ |
| `bitwise_xor(x, y)` | Bitwise XOR | ✓ | ✓ |
| `cot(value)` | Cotangent | ✓ | ✓ |
| `degrees(value)` | Radians to degrees | ✓ | ✓ |
| `radians(value)` | Degrees to radians | ✓ | ✓ |

### Date Functions

Date manipulation and extraction functions:

| Function | Description | kuzu_functions | QueryField |
|----------|-------------|----------------|------------|
| `current_date()` | Current date | ✓ | — |
| `current_timestamp()` | Current timestamp | ✓ | — |
| `date_part(part, date)` | Extract date part | ✓ | ✓ |
| `date_trunc(part, date)` | Truncate date | ✓ | ✓ |
| `datepart(part, date)` | Extract date part (alias) | ✓ | ✓ |
| `datetrunc(part, date)` | Truncate date (alias) | ✓ | ✓ |
| `dayname(date)` | Day name | ✓ | ✓ |
| `monthname(date)` | Month name | ✓ | ✓ |
| `last_day(date)` | Last day of month | ✓ | ✓ |
| `greatest(...)` | Greatest of values | ✓ | ✓ |
| `least(...)` | Least of values | ✓ | ✓ |
| `make_date(year, month, day)` | Create date | ✓ | ✓ |

### Timestamp Functions

Timestamp manipulation and extraction functions:

| Function | Description | kuzu_functions | QueryField |
|----------|-------------|----------------|------------|
| `century(timestamp)` | Extract century | ✓ | ✓ |
| `epoch_ms(ms)` | Convert milliseconds to timestamp | ✓ | — |
| `to_epoch_ms(timestamp)` | Convert timestamp to milliseconds | ✓ | ✓ |


### Interval Functions

Interval manipulation and conversion functions:

| Function | Description | kuzu_functions | QueryField |
|----------|-------------|----------------|------------|
| `to_years(value)` | Convert integer to year interval | ✓ | ✓ |
| `to_months(value)` | Convert integer to month interval | ✓ | ✓ |
| `to_days(value)` | Convert integer to day interval | ✓ | ✓ |
| `to_hours(value)` | Convert integer to hour interval | ✓ | ✓ |
| `to_minutes(value)` | Convert integer to minute interval | ✓ | ✓ |
| `to_seconds(value)` | Convert integer to second interval | ✓ | ✓ |
| `to_milliseconds(value)` | Convert integer to millisecond interval | ✓ | ✓ |
| `to_microseconds(value)` | Convert integer to microsecond interval | ✓ | ✓ |


### Timestamp Functions

Timestamp manipulation and extraction functions (order matches kuzu_functions.py):

| Function | Description | kuzu_functions | QueryField |
|----------|-------------|----------------|------------|
| `century(timestamp)` | Extract century | ✓ | ✓ |
| `epoch_ms(ms)` | Convert milliseconds to timestamp | ✓ | — |
| `to_epoch_ms(timestamp)` | Convert timestamp to milliseconds | ✓ | ✓ |



### Interval Functions

Interval manipulation and conversion functions:


### Map Functions

Map manipulation and access functions:

| Function | Description | kuzu_functions | QueryField |
|----------|-------------|----------------|------------|
| `map_func(keys, values)` | Create map from keys and values | ✓ | — |
| `map_extract(map, key)` | Extract value for key | ✓ | ✓ |
| `element_at(map, key)` | Extract value (alias) | ✓ | ✓ |
| `cardinality(map)` | Map size | ✓ | ✓ |
| `map_keys(map)` | Get all keys | ✓ | ✓ |
| `map_values(map)` | Get all values | ✓ | ✓ |

### Union Functions

Union type manipulation functions:

| Function | Description | kuzu_functions | QueryField |
|----------|-------------|----------------|------------|
| `union_value(tag := value)` | Create union with tag/value | ✓ | — |
| `union_tag(union)` | Get union tag | ✓ | ✓ |
| `union_extract(union, tag)` | Extract value for tag | ✓ | ✓ |

### Node/Relationship Functions

Node and relationship introspection functions:

| Function | Description | kuzu_functions | QueryField |
|----------|-------------|----------------|------------|
| `id_func(node_or_rel)` | Internal ID | ✓ | ✓ |
| `label(node_or_rel)` | Label name | ✓ | ✓ |
| `labels(node_or_rel)` | Label name (alias) | ✓ | ✓ |
| `offset(node_or_rel)` | ID offset | ✓ | ✓ |

### Recursive Relationship Functions

Recursive path and traversal functions:

| Function | Description | kuzu_functions | QueryField |
|----------|-------------|----------------|------------|
| `nodes(path)` | Get nodes from path | ✓ | ✓ |
| `rels(path)` | Get relationships from path | ✓ | ✓ |
| `properties(path, property)` | Get property from collection | ✓ | ✓ |
| `is_trail(path)` | Path is trail (repeated rels) | ✓ | ✓ |
| `is_acyclic(path)` | Path is acyclic (no repeated nodes) | ✓ | ✓ |
| `length(path)` | Path length (number of rels) | ✓ | ✓ |
| `cost(path)` | Weighted path cost | ✓ | ✓ |

### Array Functions

Array-specific mathematical and similarity functions:

| Function | Description | kuzu_functions | QueryField |
|----------|-------------|----------------|------------|
| `array_value(...)` | Construct array | ✓ | — |
| `array_distance(array1, array2)` | Euclidean distance | ✓ | ✓ |
| `array_squared_distance(array1, array2)` | Squared distance | ✓ | ✓ |
| `array_dot_product(array1, array2)` | Dot product | ✓ | ✓ |
| `array_inner_product(array1, array2)` | Inner product | ✓ | ✓ |
| `array_cross_product(array1, array2)` | Cross product | ✓ | ✓ |
| `array_cosine_similarity(array1, array2)` | Cosine similarity | ✓ | ✓ |


### Blob Functions

Binary data manipulation functions:

| Function | Description | kuzu_functions | QueryField |
|----------|-------------|----------------|------------|
| `blob(data)` | Create blob | ✓ | ✓ |
| `encode(data)` | Encode string to blob | ✓ | ✓ |
| `decode(blob)` | Decode blob to string | ✓ | ✓ |
| `octet_length(blob)` | Blob byte length | ✓ | ✓ |

### Struct Functions

Struct manipulation functions:

| Function | Description | kuzu_functions | QueryField |
|----------|-------------|----------------|------------|
| `struct_extract(struct, field)` | Extract struct field | ✓ | ✓ |

| `to_int32(value)` | Cast to int32 | `ka.to_int32(field)` |
| `to_int16(value)` | Cast to int16 | `ka.to_int16(field)` |
| `to_float(value)` | Cast to float | `ka.to_float(field)` |
| `to_date(value)` | Cast to date | `ka.to_date(field)` |
| `to_timestamp(value)` | Cast to timestamp | `ka.to_timestamp(field)` |
| `cast_as(value, type)` | Cast using AS syntax | `ka.cast_as(field, "INT64")` |
| `case([input])` | CASE expression | `ka.case()` |

### Hash Functions

Cryptographic hash functions:

| Function | Description | kuzu_functions | QueryField |
|----------|-------------|----------------|------------|
| `md5(data)` | MD5 hash | ✓ | ✓ |
| `sha256(data)` | SHA256 hash | ✓ | ✓ |
| `hash(data)` | Generic hash | ✓ | ✓ |

### UUID Functions

UUID generation and manipulation functions:

| Function | Description | kuzu_functions | QueryField |
|----------|-------------|----------------|------------|
| `gen_random_uuid()` | Generate random UUID | ✓ | — |
| `uuid(string)` | Parse UUID string | ✓ | ✓ |

### Utility Functions

Utility and miscellaneous functions:

| Function | Description | kuzu_functions | QueryField |
|----------|-------------|----------------|------------|
| `coalesce(val1, val2, ...)` | First non-NULL value | ✓ | ✓ |
| `ifnull(value, replacement)` | Replace NULL with value | ✓ | ✓ |
| `nullif(a, b)` | NULL if equal | ✓ | ✓ |
| `typeof(value)` | Get value type | ✓ | ✓ |
| `constant_or_null(constant, check)` | Constant if check non-NULL | ✓ | ✓ |
| `count_if(condition)` | 1 if condition true else 0 | ✓ | ✓ |
| `error(message)` | Raise runtime error | ✓ | ✓ |


### Casting Functions

Type conversion and casting functions:

|----------|-------------|----------------|------------|
| `to_int64(value)` | Cast to INT64 | ✓ | — |
| `to_int32(value)` | Cast to INT32 | ✓ | — |
| `to_int16(value)` | Cast to INT16 | ✓ | — |
| `to_double(value)` | Cast to DOUBLE | ✓ | — |
| `to_float(value)` | Cast to FLOAT | ✓ | — |
| `to_string(value)` | Cast to STRING | ✓ | — |
| `to_date(value)` | Cast to DATE | ✓ | — |
| `to_timestamp(value)` | Cast to TIMESTAMP | ✓ | ✓ |
| `cast(value, type)` | CAST function | ✓ | ✓ |
| `cast_as(value, type)` | CAST AS syntax | ✓ | ✓ |
| `case([input])` | Create CASE expression | ✓ | ✓ |




## Operator Reference

KuzuAlchemy operator support (exactly as implemented):

### Comparison Operators

| Operator/Method | Description | Backing Enum/Method |
|-----------------|-------------|---------------------|
| `==` | Equal to | QueryField.__eq__ -> ComparisonOperator.EQ |
| `!=` | Not equal | QueryField.__ne__ -> ComparisonOperator.NEQ |
| `<` | Less than | QueryField.__lt__ -> ComparisonOperator.LT |
| `<=` | Less than or equal | QueryField.__le__ -> ComparisonOperator.LTE |
| `>` | Greater than | QueryField.__gt__ -> ComparisonOperator.GT |
| `>=` | Greater than or equal | QueryField.__ge__ -> ComparisonOperator.GTE |
| `in_(values)` | Membership | QueryField.in_ -> ComparisonOperator.IN |
| `not_in(values)` | Not in | QueryField.not_in -> ComparisonOperator.NOT_IN |
| `between(a,b, inclusive=True)` | Range test | QueryField.between -> BetweenExpression |

### Pattern/Regex Operators

| Operator/Method | Description | Backing Enum |
|-----------------|-------------|--------------|
| `like(pattern, case_sensitive=True)` | Regex-like match | ComparisonOperator.LIKE |
| `not_like(pattern, case_sensitive=True)` | Negative match | ComparisonOperator.NOT_LIKE |
| `regex_match(pattern)` | `=~` regex match | ComparisonOperator.REGEX_MATCH |
| `not_regex_match(pattern)` | `!~` negative regex | ComparisonOperator.NOT_REGEX_MATCH |

### Contains/Prefix/Suffix Filters

| Method | Description | Backing Enum |
|--------|-------------|--------------|
| `contains_filter(value)` | Contains element/substr | ComparisonOperator.CONTAINS |
| `starts_with_filter(value, case_sensitive=True)` | Prefix match | ComparisonOperator.STARTS_WITH |
| `ends_with_filter(value, case_sensitive=True)` | Suffix match | ComparisonOperator.ENDS_WITH |
| `is_null()` | Field is NULL | ComparisonOperator.IS_NULL |
| `is_not_null()` | Field is NOT NULL | ComparisonOperator.IS_NOT_NULL |

### Logical Operators (on FilterExpression)

| Operator | Description | Backing |
|----------|-------------|---------|
| `&` | Logical AND | FilterExpression.__and__ -> LogicalOperator.AND |
| `|` | Logical OR | FilterExpression.__or__ -> LogicalOperator.OR |
| `^` | Logical XOR | FilterExpression.__xor__ -> LogicalOperator.XOR |
| `~` | Logical NOT | FilterExpression.__invert__ -> NotFilterExpression |

### Arithmetic Operators (on QueryField)

| Operator | Description | Backing |
|----------|-------------|---------|
| `+` | Addition / list concatenation | QueryField.__add__/__radd__ -> ArithmeticOperator.ADD |
| `-` | Subtraction | QueryField.__sub__/__rsub__ -> ArithmeticOperator.SUB |
| `*` | Multiplication | QueryField.__mul__/__rmul__ -> ArithmeticOperator.MUL |
| `/` | Division | QueryField.__truediv__/__rtruediv__ -> ArithmeticOperator.DIV |
| `%` | Modulo | QueryField.__mod__/__rmod__ -> ArithmeticOperator.MOD |
| `^` | Power | QueryField.__pow__/__rpow__ -> ArithmeticOperator.POW |

### Indexing/Slicing (on QueryField)

| Operator | Description | Backing Function |
|----------|-------------|------------------|
| `field[idx]` | 1-based index extract | FunctionExpression("array_extract") |
| `field[a:b]` | 1-based slice | FunctionExpression("array_slice") |

---

## Model Definition

### Node Models

```python
from kuzualchemy import kuzu_node, KuzuBaseModel, kuzu_field, KuzuDataType
from typing import Optional, List
from datetime import datetime

@kuzu_node("User")  # Table name in Kuzu
class User(KuzuBaseModel):
    # Primary key
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)

    # Basic fields
    name: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    email: Optional[str] = kuzu_field(kuzu_type=KuzuDataType.STRING, unique=True, default=None)
    age: int = kuzu_field(kuzu_type=KuzuDataType.INT32, default=0)

    # Boolean fields
    is_active: bool = kuzu_field(kuzu_type=KuzuDataType.BOOL, default=True)

    # Timestamp fields
    created_at: datetime = kuzu_field(
        kuzu_type=KuzuDataType.TIMESTAMP,
        default=KuzuDefaultFunction.CURRENT_TIMESTAMP
    )

    # Array fields
    tags: Optional[List[str]] = kuzu_field(
        kuzu_type=ArrayTypeSpecification(element_type=KuzuDataType.STRING),
        default=None
    )
```

### Relationship Models

```python
from kuzualchemy import kuzu_relationship, KuzuRelationshipBase

@kuzu_relationship("KNOWS", pairs=[(User, User)])
class Knows(KuzuRelationshipBase):
    since: datetime = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP)
    strength: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, default=1.0)
```

### Model Methods

Every model inherits these methods from `KuzuBaseModel`:

```python
class User(KuzuBaseModel):
    # Built-in methods available:

    def save(self, session: KuzuSession) -> None:
        """Save instance to database"""
        pass

    def delete(self, session: KuzuSession) -> None:
        """Delete instance from database"""
        pass

    @classmethod
    def query(cls, session: KuzuSession = None) -> Query:
        """Create query for this model"""
        pass

    @classmethod
    def get_primary_key_fields(cls) -> List[str]:
        """Get primary key field names"""
        pass

    @classmethod
    def get_foreign_key_fields(cls) -> Dict[str, ForeignKeyReference]:
        """Get foreign key fields"""
        pass
```

---

## Field Types & Metadata

### Supported Kuzu Data Types

```python
from kuzualchemy import KuzuDataType

# Numeric types
KuzuDataType.INT8, KuzuDataType.INT16, KuzuDataType.INT32, KuzuDataType.INT64
KuzuDataType.UINT8, KuzuDataType.UINT16, KuzuDataType.UINT32, KuzuDataType.UINT64
KuzuDataType.FLOAT, KuzuDataType.DOUBLE
KuzuDataType.DECIMAL, KuzuDataType.SERIAL

# String types
KuzuDataType.STRING, KuzuDataType.BLOB

# Boolean type
KuzuDataType.BOOL

# Date/time types
KuzuDataType.DATE, KuzuDataType.TIMESTAMP, KuzuDataType.INTERVAL

# UUID type
KuzuDataType.UUID

# Complex types
KuzuDataType.STRUCT, KuzuDataType.MAP, KuzuDataType.UNION
```

### Field Definition

```python
# Field definition
field = kuzu_field(
    # Basic properties
    kuzu_type=KuzuDataType.STRING,
    primary_key=False,
    unique=False,
    not_null=False,
    index=False,

    # Default values
    default="default_value",
    default_factory=lambda: "computed_default",

    # Constraints
    check_constraint="LENGTH(field_name) > 0",

    # Foreign keys
    foreign_key=ForeignKeyReference(
        target_model=TargetModel,
        target_field="id",
        on_delete=CascadeAction.CASCADE,
        on_update=CascadeAction.SET_NULL
    ),

    # Metadata
    alias="field_alias",
    title="Field Title",
    description="Field description"
)
```

### Array Fields

```python
from kuzualchemy.kuzu_orm import ArrayTypeSpecification

class User(KuzuBaseModel):
    # Array field definition
    tags: List[str] = kuzu_field(
        kuzu_type=ArrayTypeSpecification(element_type=KuzuDataType.STRING),
        default=None
    )
```

### Default Values

```python
from kuzualchemy.constants import KuzuDefaultFunction

class User(KuzuBaseModel):
    # Static defaults
    status: str = kuzu_field(kuzu_type=KuzuDataType.STRING, default="active")

    # Function defaults
    created_at: datetime = kuzu_field(
        kuzu_type=KuzuDataType.TIMESTAMP,
        default=KuzuDefaultFunction.CURRENT_TIMESTAMP
    )

    # Factory defaults
    uuid_field: str = kuzu_field(
        kuzu_type=KuzuDataType.UUID,
        default_factory=lambda: str(uuid.uuid4())
    )
```

---

## Relationships

### Basic Relationships

```python
from kuzualchemy import kuzu_relationship, KuzuRelationshipBase

@kuzu_relationship("FOLLOWS", pairs=[(User, User)])
class Follows(KuzuRelationshipBase):
    since: datetime = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP)
    weight: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, default=1.0)
```

### Multi-Pair Relationships

```python
# Multiple relationship pairs
@kuzu_relationship("AUTHORED", pairs=[
    (User, {Post, Comment}),
    (Organization, Post)
])
class Authored(KuzuRelationshipBase):
    created_at: datetime = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP)
    role: str = kuzu_field(kuzu_type=KuzuDataType.STRING, default="author")
```

### Relationship Usage

```python
# Create relationships
user1 = User(id=1, name="Alice")
user2 = User(id=2, name="Bob")
follows = Follows(from_node=user1, to_node=user2, since=datetime.now())

session.add_all([user1, user2, follows])
session.commit()
```

---

## Query System

### Basic Queries

```python
from kuzualchemy import Query

# Create query
query = Query(User, session=session)

# Simple filtering
filtered = query.where(query.fields.age > 25)
results = filtered.all()

# Method chaining
results = (Query(User, session=session)
    .where(Query(User, session=session).fields.name.starts_with("A"))
    .where(Query(User, session=session).fields.age.between(20, 40))
    .order_by(Query(User, session=session).fields.name.asc())
    .limit(10)
    .all())
```

### Advanced Queries

```python
# Aggregation with HAVING: count users by age > 1
q = Query(User, session=session)
agg = q.count()  # COUNT(*) AS count
count_by_age = (
    q.group_by(q.fields.age)
     .having(ka.to_int64("count") > 1)  # compare aggregated alias post-WITH using cast
)

# Relationship join (pattern: join(TargetModel, RelationshipClass, ...))
q = Query(User, session=session)
joined = q.join(User, Follows, target_alias="u2")

# Subquery: authors older than 30
subq = Query(User, session=session).where(Query(User, session=session).fields.age > 30)
main_query = Query(Post, session=session).where(
    Query(Post, session=session).fields.author_id.in_(subq.select("id"))
)
```

### Function Usage in Queries

```python
import kuzualchemy as ka

# Text functions
query = Query(User, session=session)
text_query = query.where(
    ka.upper(query.fields.name).starts_with("A")
)

# Numeric functions
numeric_query = query.where(
    ka.abs(query.fields.age - 30) < 5
)

# List functions
list_query = query.where(
    ka.list_contains(query.fields.hobbies, "reading")
)

# Date functions
date_query = query.where(
    ka.date_part("year", query.fields.birth_date) > 1990
)
```

### Query Results

```python
# Get all results
results = query.all()  # List[ModelType]

# Get first result
first = query.first()  # ModelType | None

# Get exactly one result
one = query.one()  # ModelType (raises if 0 or >1)

# Get one or none
one_or_none = query.one_or_none()  # ModelType | None

# Check existence
exists = query.exists()  # bool

# Count results
count_query = query.count()         # Query with COUNT(*) AS count aggregation
count = count_query._execute()[0]["count"] if count_query._execute() else 0
```

---

## Session Management

### Basic Session Usage

```python
from kuzualchemy import KuzuSession
from pathlib import Path

# Create session
session = KuzuSession(db_path=Path("my_database.db"))

# Execute DDL
ddl = get_all_ddl()
if ddl.strip():
    session.execute(ddl)

# Add and commit
user = User(id=1, name="Alice", email="alice@example.com")
session.add(user)
session.commit()

# Close session
session.close()
```

### Transaction Management

```python
# Manual transactions using KuzuTransaction
from kuzualchemy import KuzuTransaction

with KuzuTransaction(session):
    user = User(id=1, name="Alice")
    session.add(user)
    # Automatic commit on success, rollback on exception

# Or using session.begin() context manager
with session.begin():
    user = User(id=1, name="Alice")
    session.add(user)
    # Automatic commit on success, rollback on exception
```

### Session Factory

```python
from kuzualchemy import SessionFactory

# Create factory
factory = SessionFactory(
    db_path="database.db",
    autoflush=True,
    autocommit=False
)

# Create sessions
session1 = factory.create_session()
session2 = factory.create_session(autocommit=True)  # Override defaults

# Session scope context manager
with factory.session_scope() as session:
    user = User(id=1, name="Alice")
    session.add(user)
    # Automatic commit/rollback
```

### Connection Management

```python
from kuzualchemy import KuzuConnection

# Direct connection usage
connection = KuzuConnection(db_path="database.db")
session = KuzuSession(connection=connection)

# Connection sharing
session1 = KuzuSession(connection=connection)
session2 = KuzuSession(connection=connection)
```

---

## Advanced Features

### Registry Management

```python
from kuzualchemy import (
    get_registered_nodes,
    get_registered_relationships,
    get_all_models,
    clear_registry,
    validate_all_models
)

# Access registered models
nodes = get_registered_nodes()
relationships = get_registered_relationships()
all_models = get_all_models()

# Validate all models
validation_errors = validate_all_models()

# Clear registry (useful for testing)
clear_registry()
```

### Enhanced Base Model with Enum Conversion

```python
from kuzualchemy import BaseModel
from enum import Enum

class Status(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"

@kuzu_node("Account")
class Account(BaseModel):  # Automatic enum conversion
    status: Status = kuzu_field(kuzu_type=KuzuDataType.STRING)

    # BaseModel automatically converts enums to/from string values
```

### Foreign Key Support

```python
from kuzualchemy import ForeignKeyReference, CascadeAction

@kuzu_node("Post")
class Post(KuzuBaseModel):
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    title: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
    author_id: int = kuzu_field(
        kuzu_type=KuzuDataType.INT64,
        foreign_key=ForeignKeyReference(
            target_model=User,
            target_field="id",
            on_delete=CascadeAction.CASCADE
        )
    )
```

### Custom Functions

```python
# All Kuzu functions are available as standalone callables
import kuzualchemy as ka

# Use in queries
query = Query(User, session=session).where(
    ka.concat(query.fields.first_name, " ", query.fields.last_name).contains("Alice")
)

# Use in expressions
full_name = ka.concat(user.first_name, " ", user.last_name)
```

### Complex Expressions

```python
# Combine multiple functions and operators
complex_filter = (
    ka.upper(query.fields.name).starts_with("A") &
    (query.fields.age.between(20, 40)) &
    ka.list_contains(query.fields.tags, "python")
)

results = Query(User, session=session).where(complex_filter).all()
```

---

## API Reference

### Core Classes

#### KuzuBaseModel
Base class for all node models with built-in ORM functionality.

**Methods:**
- `save(session: KuzuSession) -> None`: Save instance to database
- `delete(session: KuzuSession) -> None`: Delete instance from database
- `query(session: KuzuSession = None) -> Query`: Create query for this model
- `get_kuzu_metadata(field_name: str) -> KuzuFieldMetadata`: Get field metadata
- `get_all_kuzu_metadata() -> Dict[str, KuzuFieldMetadata]`: Get all field metadata
- `get_primary_key_fields() -> List[str]`: Get primary key field names
- `get_foreign_key_fields() -> Dict[str, ForeignKeyReference]`: Get foreign key fields

#### KuzuRelationshipBase
Base class for relationship models.

**Methods:**
- Same as KuzuBaseModel plus relationship-specific functionality
- `create_between(from_node, to_node, **properties) -> KuzuRelationshipBase`: Factory to instantiate relationship between nodes
- `from_node_pk`/`to_node_pk` properties for node primary keys

#### KuzuSession
Main session class for database operations.

**Methods:**
- `execute(query: str, parameters: Dict = None) -> List[Dict]`: Execute raw query
- `add(instance: Any) -> None`: Add instance to session
- `add_all(instances: List[Any]) -> None`: Add multiple instances
- `delete(instance: Any) -> None`: Mark instance for deletion
- `commit() -> None`: Commit current transaction
- `rollback() -> None`: Rollback current transaction
- `flush() -> None`: Flush pending changes
- `close() -> None`: Close session
- `begin() -> KuzuTransaction`: Begin transaction context

#### Query[ModelType]
Type-safe query builder.

**Methods:**
- `where(expression: FilterExpression) -> Query`: Add WHERE clause
- `filter(*expressions: FilterExpression) -> Query`: Add multiple filters
- `order_by(*fields: QueryField) -> Query`: Add ORDER BY clause
- `group_by(*fields: QueryField) -> Query`: Add GROUP BY clause
- `having(expression: FilterExpression) -> Query`: Add HAVING clause
- `limit(count: int) -> Query`: Add LIMIT clause
- `offset(count: int) -> Query`: Add OFFSET clause
- `distinct() -> Query`: Add DISTINCT clause
- `all() -> List[ModelType]`: Execute and return all results
- `first() -> ModelType | None`: Execute and return first result
- `one() -> ModelType`: Execute and return exactly one result
- `one_or_none() -> ModelType | None`: Execute and return one or none
- `count() -> int`: Count results
- `exists() -> bool`: Check if results exist

### Field Definition

#### kuzu_field Function
Field definition with options:

```python
kuzu_field(
    default: Any = ...,                                    # Default value
    kuzu_type: Union[KuzuDataType, str, ArrayTypeSpecification], # Kuzu data type
    primary_key: bool = False,                            # Primary key flag
    foreign_key: ForeignKeyReference = None,               # Foreign key metadata
    unique: bool = False,                                 # Unique constraint
    not_null: bool = False,                              # NOT NULL constraint
    index: bool = False,                                 # Index flag
    check_constraint: str = None,                        # CHECK constraint
    default_factory: Callable[[], Any] = None,          # Default factory function
    alias: str = None,                                   # Field alias
    title: str = None,                                   # Field title
    description: str = None,                             # Field description
)
```

### Decorators

#### @kuzu_node()
Mark class as Kuzu node:

```python
@kuzu_node(
    name: str = None,                                    # Node name (defaults to class name)
    abstract: bool = False,                              # Abstract node flag
    compound_indexes: List[CompoundIndex] = None,        # Compound indexes
    table_constraints: List[str] = None,                 # Table constraints
    properties: Dict[str, Any] = None                    # Additional properties
)
```

#### @kuzu_relationship()
Mark class as Kuzu relationship:

```python
@kuzu_relationship(
    name: str = None,                                    # Relationship name
    pairs: List[Tuple[Type, Type]] = None,              # Valid node pairs
    multiplicity: RelationshipMultiplicity = MANY_TO_MANY, # Relationship multiplicity
    compound_indexes: List[CompoundIndex] = None,        # Compound indexes
    table_constraints: List[str] = None,                 # Table constraints
    properties: Dict[str, Any] = None                    # Additional properties
)
```

### Enums and Constants

#### KuzuDataType
All supported Kuzu data types:
- Numeric: `INT8`, `INT16`, `INT32`, `INT64`, `UINT8`, `UINT16`, `UINT32`, `UINT64`, `FLOAT`, `DOUBLE`, `DECIMAL`, `SERIAL`
- String: `STRING`, `BLOB`
- Boolean: `BOOL`
- Temporal: `DATE`, `TIMESTAMP`, `INTERVAL`
- Other: `UUID`, `STRUCT`, `MAP`, `UNION`

#### ComparisonOperator
Query comparison operators (from kuzu_query_expressions):
- `EQ`, `NEQ`: Equality/inequality
- `LT`, `LTE`, `GT`, `GTE`: Comparison operators
- `IN`, `NOT_IN`: List membership
- `LIKE`, `NOT_LIKE`: Pattern matching
- `IS_NULL`, `IS_NOT_NULL`: Null checks
- `CONTAINS`: String/array contains
- `STARTS_WITH`, `ENDS_WITH`: String prefix/suffix
- `EXISTS`, `NOT_EXISTS`: Existence checks

#### LogicalOperator
Logical operators for combining conditions:
- `AND`, `OR`, `NOT`, `XOR`: Boolean logic

#### AggregateFunction
Aggregate functions:
- `COUNT`, `COUNT_DISTINCT`: Counting
- `SUM`, `AVG`: Numeric aggregation
- `MIN`, `MAX`: Extrema
- `COLLECT`, `COLLECT_LIST`, `COLLECT_SET`: Collection aggregation

#### OrderDirection
Ordering directions:
- `ASC`: Ascending order
- `DESC`: Descending order

#### JoinType
Join types:
- `INNER`: Inner join
- `OPTIONAL`: Optional match (left outer join)

### Utility Functions

#### DDL Generation
- `get_all_ddl() -> str`: Generate DDL for all registered models
- `get_ddl_for_node(node_cls: Type[Any]) -> str`: Generate DDL for specific node
- `get_ddl_for_relationship(rel_cls: Type[Any]) -> str`: Generate DDL for specific relationship

#### Registry Management
- `get_registered_nodes() -> Dict[str, Type[Any]]`: Get all registered nodes
- `get_registered_relationships() -> Dict[str, Type[Any]]`: Get all registered relationships
- `get_all_models() -> Dict[str, Type[Any]]`: Get all registered models
- `clear_registry() -> None`: Clear model registry
- `validate_all_models() -> List[str]`: Validate all registered models

#### Test Utilities
- `initialize_schema(session: KuzuSession, ddl: str = None) -> None`: Initialize database schema

---

---

## Contributing

We welcome contributions to KuzuAlchemy! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd kuzualchemy

# Install in development mode
pip install -e ".[dev,test]"

# Run tests
pytest

# Run type checking
mypy src/

# Run linting
flake8 src/
black src/

# Build package
python -m build
```

### Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_functions.py
pytest tests/test_integration.py

# Run with coverage
pytest --cov=kuzualchemy --cov-report=html
```

---

## License

This project is licensed under the Apache 2.0 license - see the [LICENSE](LICENSE) file for details.

---

## Conclusion

KuzuAlchemy is an Object-Relational Mapping library for the Kuzu graph database. It provides a SQLAlchemy-like interface for working with graph data.

## Claims

Kuzualchemy™ is a trademark claimed by FanaticPythoner.


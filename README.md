# KuzuAlchemy: A SQLAlchemy-like ORM for Kuzu graph database
# Version: 0.1.0
**Status**: Alpha

[![PyPI version](https://badge.fury.io/py/kuzualchemy.svg)](https://badge.fury.io/py/kuzualchemy)
[![Python versions](https://img.shields.io/pypi/pyversions/kuzualchemy.svg)](https://pypi.org/project/kuzualchemy/)
[![Tests](https://github.com/kuzualchemy/kuzualchemy/workflows/Tests/badge.svg)](https://github.com/kuzualchemy/kuzualchemy/actions)
[![Coverage](https://codecov.io/gh/kuzualchemy/kuzualchemy/branch/main/graph/badge.svg)](https://codecov.io/gh/kuzualchemy/kuzualchemy)

KuzuAlchemy is an Object-Relational Mapping (ORM) library for the [Kuzu graph database](https://kuzudb.com/). It provides a SQLAlchemy-like interface for working with graph data.

> **Note**: This software is currently in alpha development. APIs may change, and it should not be used in production environments.

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

**⚠️ Warning**: This is alpha software. Use at your own risk and do not use in production.

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

## Quick Start

### Basic Setup

```python
from kuzualchemy import (
    KuzuBaseModel, KuzuRelationshipBase,
    node, relationship, Field,
    KuzuDataType, KuzuSession,
    get_all_ddl
)

# Create session
session = KuzuSession(db_path="database.db")

# Initialize schema
ddl = get_all_ddl()
if ddl.strip():
    from kuzualchemy.test_utilities import initialize_schema
    initialize_schema(session)
```

### Example

```python
import kuzualchemy as ka
from pathlib import Path

# Define your graph models
@ka.node("Person")
class Person(ka.KuzuBaseModel):
    name: str = ka.Field(primary_key=True)
    age: int
    email: str

@ka.relationship("KNOWS")
class Knows(ka.KuzuBaseModel):
    since: int
    strength: float = 1.0

# Create database and session
db_path = Path("my_graph.db")
session = ka.KuzuSession(db_path)

# Create schema
session.execute_ddl(ka.get_all_ddl())

# Insert data
alice = Person(name="Alice", age=30, email="alice@example.com")
bob = Person(name="Bob", age=25, email="bob@example.com")
knows = Knows(since=2020, strength=0.9)

session.add(alice)
session.add(bob)
session.add_relationship(alice, knows, bob)
session.commit()

# Query data
query = ka.Query(Person, session=session)
filtered_query = query.where(query.fields.age > 25)
results = filtered_query.all()

print(f"Found {len(results)} people over 25")
```

---

## Function Reference

KuzuAlchemy implements Kuzu functions across multiple categories. Each function is available both as a standalone callable and as a QueryField method for fluent API usage.

### Text Functions

String manipulation and text processing functions:

| Function | Description | Implementation |
|----------|-------------|----------------|
| `concat(str1, str2, ...)` | Concatenate strings | `kuzu_functions.py:15` |
| `upper(string)` | Convert to uppercase | `kuzu_functions.py:20` |
| `lower(string)` | Convert to lowercase | `kuzu_functions.py:25` |
| `substring(string, start, length)` | Extract substring | `kuzu_functions.py:30` |
| `trim(string)` | Remove whitespace | `kuzu_functions.py:35` |
| `ltrim(string)` | Remove left whitespace | `kuzu_functions.py:40` |
| `rtrim(string)` | Remove right whitespace | `kuzu_functions.py:45` |
| `lpad(string, length, fill)` | Left pad string | `kuzu_functions.py:50` |
| `rpad(string, length, fill)` | Right pad string | `kuzu_functions.py:55` |
| `repeat(string, count)` | Repeat string | `kuzu_functions.py:60` |
| `reverse(string)` | Reverse string | `kuzu_functions.py:65` |
| `replace(string, search, replace)` | Replace substring | `kuzu_functions.py:70` |
| `split(string, delimiter)` | Split string | `kuzu_functions.py:75` |
| `array_to_string(array, delimiter)` | Join array to string | `kuzu_functions.py:80` |
| `string_to_array(string, delimiter)` | Split string to array | `kuzu_functions.py:85` |
| `starts_with(string, prefix)` | Check prefix | `kuzu_functions.py:90` |
| `ends_with(string, suffix)` | Check suffix | `kuzu_functions.py:95` |
| `contains(string, substring)` | Check contains | `kuzu_functions.py:100` |
| `length(string)` | String length | `kuzu_functions.py:105` |
| `char_length(string)` | Character length | `kuzu_functions.py:110` |
| `bit_length(string)` | Bit length | `kuzu_functions.py:115` |
| `octet_length(string)` | Byte length | `kuzu_functions.py:120` |
| `left(string, length)` | Left substring | `kuzu_functions.py:125` |
| `right(string, length)` | Right substring | `kuzu_functions.py:130` |
| `ascii(string)` | ASCII value | `kuzu_functions.py:135` |
| `chr(code)` | Character from code | `kuzu_functions.py:140` |
| `initcap(string)` | Initial caps | `kuzu_functions.py:145` |
| `title(string)` | Title case | `kuzu_functions.py:150` |
| `position(substring, string)` | Find position | `kuzu_functions.py:155` |
| `strpos(string, substring)` | Find position | `kuzu_functions.py:160` |
| `encode(string, format)` | Encode string | `kuzu_functions.py:165` |

### List Functions

Array and list manipulation functions:

| Function | Description | Implementation |
|----------|-------------|----------------|
| `list_creation(...)` | Create list | `kuzu_functions.py:274` |
| `list_extract(list, index)` | Extract element | `kuzu_functions.py:279` |
| `list_element(list, index)` | Get element | `kuzu_functions.py:284` |
| `array_extract(array, index)` | Extract from array | `kuzu_functions.py:289` |
| `list_len(list)` | List length | `kuzu_functions.py:294` |
| `array_length(array)` | Array length | `kuzu_functions.py:299` |
| `size(list)` | Collection size | `kuzu_functions.py:304` |
| `list_concat(list1, list2)` | Concatenate lists | `kuzu_functions.py:309` |
| `array_concat(array1, array2)` | Concatenate arrays | `kuzu_functions.py:314` |
| `list_cat(list1, list2)` | Concatenate lists | `kuzu_functions.py:319` |
| `array_cat(array1, array2)` | Concatenate arrays | `kuzu_functions.py:324` |
| `list_append(list, element)` | Append element | `kuzu_functions.py:329` |
| `array_append(array, element)` | Append to array | `kuzu_functions.py:334` |
| `list_prepend(element, list)` | Prepend element | `kuzu_functions.py:339` |
| `array_prepend(element, array)` | Prepend to array | `kuzu_functions.py:344` |
| `list_position(list, element)` | Find position | `kuzu_functions.py:349` |
| `array_position(array, element)` | Find in array | `kuzu_functions.py:354` |
| `list_contains(list, element)` | Check contains | `kuzu_functions.py:359` |
| `array_contains(array, element)` | Check array contains | `kuzu_functions.py:364` |
| `list_slice(list, start, end)` | Slice list | `kuzu_functions.py:369` |
| `array_slice(array, start, end)` | Slice array | `kuzu_functions.py:374` |
| `list_sort(list)` | Sort list | `kuzu_functions.py:379` |
| `array_sort(array)` | Sort array | `kuzu_functions.py:384` |
| `list_reverse_sort(list)` | Reverse sort | `kuzu_functions.py:389` |
| `list_sum(list)` | Sum elements | `kuzu_functions.py:394` |
| `list_product(list)` | Product elements | `kuzu_functions.py:399` |
| `list_min(list)` | Minimum element | `kuzu_functions.py:404` |
| `list_max(list)` | Maximum element | `kuzu_functions.py:409` |
| `list_avg(list)` | Average elements | `kuzu_functions.py:414` |
| `list_distinct(list)` | Distinct elements | `kuzu_functions.py:419` |
| `list_unique(list)` | Unique elements | `kuzu_functions.py:424` |
| `list_any_value(list)` | Any element | `kuzu_functions.py:429` |
| `list_reduce(list, initial, func)` | Reduce list | `kuzu_functions.py:434` |
| `range(start, end)` | Generate range | `kuzu_functions.py:439` |
| `list_has_any(list1, list2)` | Check overlap | `kuzu_functions.py:444` |
| `list_has_all(list1, list2)` | Check contains all | `kuzu_functions.py:449` |
| `list_zip(list1, list2)` | Zip lists | `kuzu_functions.py:454` |
| `list_transform(list, func)` | Transform elements | `kuzu_functions.py:459` |
| `list_filter(list, predicate)` | Filter elements | `kuzu_functions.py:464` |
| `list_aggregate(list, func)` | Aggregate list | `kuzu_functions.py:469` |
| `flatten(list)` | Flatten nested | `kuzu_functions.py:474` |
| `unnest(list)` | Unnest list | `kuzu_functions.py:479` |

### Numeric Functions

Mathematical and numeric computation functions:

| Function | Description | Implementation |
|----------|-------------|----------------|
| `abs(number)` | Absolute value | `kuzu_functions.py:173` |
| `acos(number)` | Arc cosine | `kuzu_functions.py:178` |
| `asin(number)` | Arc sine | `kuzu_functions.py:183` |
| `atan(number)` | Arc tangent | `kuzu_functions.py:188` |
| `atan2(y, x)` | Arc tangent of y/x | `kuzu_functions.py:193` |
| `ceil(number)` | Ceiling | `kuzu_functions.py:198` |
| `ceiling(number)` | Ceiling | `kuzu_functions.py:203` |
| `cos(number)` | Cosine | `kuzu_functions.py:208` |
| `cot(number)` | Cotangent | `kuzu_functions.py:213` |
| `degrees(radians)` | Convert to degrees | `kuzu_functions.py:218` |
| `even(number)` | Check if even | `kuzu_functions.py:223` |
| `exp(number)` | Exponential | `kuzu_functions.py:228` |
| `factorial(number)` | Factorial | `kuzu_functions.py:233` |
| `floor(number)` | Floor | `kuzu_functions.py:238` |
| `gamma(number)` | Gamma function | `kuzu_functions.py:243` |
| `lgamma(number)` | Log gamma | `kuzu_functions.py:248` |
| `ln(number)` | Natural log | `kuzu_functions.py:253` |
| `log(number)` | Logarithm | `kuzu_functions.py:258` |
| `log10(number)` | Base 10 log | `kuzu_functions.py:263` |
| `log2(number)` | Base 2 log | `kuzu_functions.py:268` |
| `pi()` | Pi constant | `kuzu_functions.py:273` |
| `pow(base, exponent)` | Power | `kuzu_functions.py:278` |
| `power(base, exponent)` | Power | `kuzu_functions.py:283` |
| `radians(degrees)` | Convert to radians | `kuzu_functions.py:288` |
| `round(number, precision)` | Round number | `kuzu_functions.py:293` |
| `sign(number)` | Sign of number | `kuzu_functions.py:298` |
| `sin(number)` | Sine | `kuzu_functions.py:303` |
| `sqrt(number)` | Square root | `kuzu_functions.py:308` |
| `tan(number)` | Tangent | `kuzu_functions.py:313` |

### Date Functions

Date manipulation and extraction functions:

| Function | Description | Implementation |
|----------|-------------|----------------|
| `date_part(part, date)` | Extract date part | `kuzu_functions.py:467` |
| `datepart(part, date)` | Extract date part | `kuzu_functions.py:472` |
| `date_trunc(part, date)` | Truncate date | `kuzu_functions.py:477` |
| `datetrunc(part, date)` | Truncate date | `kuzu_functions.py:482` |
| `dayname(date)` | Day name | `kuzu_functions.py:487` |
| `monthname(date)` | Month name | `kuzu_functions.py:492` |
| `last_day(date)` | Last day of month | `kuzu_functions.py:497` |
| `make_date(year, month, day)` | Create date | `kuzu_functions.py:502` |
| `greatest(date1, date2, ...)` | Latest date | `kuzu_functions.py:507` |
| `least(date1, date2, ...)` | Earliest date | `kuzu_functions.py:512` |
| `date_diff(part, start, end)` | Date difference | `kuzu_functions.py:517` |

### Timestamp Functions

Timestamp manipulation and extraction functions:

| Function | Description | Implementation |
|----------|-------------|----------------|
| `century(timestamp)` | Extract century | `kuzu_functions.py:527` |
| `epoch_ms(timestamp)` | Milliseconds since epoch | `kuzu_functions.py:532` |
| `to_timestamp(seconds)` | Convert to timestamp | `kuzu_functions.py:537` |
| `date_part(part, timestamp)` | Extract timestamp part | `kuzu_functions.py:542` |
| `date_trunc(part, timestamp)` | Truncate timestamp | `kuzu_functions.py:547` |
| `make_timestamp(y, m, d, h, min, s)` | Create timestamp | `kuzu_functions.py:552` |
| `timestamp_diff(part, start, end)` | Timestamp difference | `kuzu_functions.py:557` |
| `timestamp_trunc(part, timestamp)` | Truncate timestamp | `kuzu_functions.py:562` |
| `extract(part FROM timestamp)` | Extract part | `kuzu_functions.py:567` |
| `age(timestamp1, timestamp2)` | Age between timestamps | `kuzu_functions.py:572` |
| `clock_timestamp()` | Current timestamp | `kuzu_functions.py:577` |
| `current_timestamp()` | Current timestamp | `kuzu_functions.py:582` |
| `now()` | Current timestamp | `kuzu_functions.py:587` |
| `transaction_timestamp()` | Transaction timestamp | `kuzu_functions.py:592` |

### Interval Functions

Interval manipulation and conversion functions:

| Function | Description | Implementation |
|----------|-------------|----------------|
| `to_years(interval)` | Convert to years | `kuzu_functions.py:624` |
| `to_months(interval)` | Convert to months | `kuzu_functions.py:629` |
| `to_days(interval)` | Convert to days | `kuzu_functions.py:634` |
| `to_hours(interval)` | Convert to hours | `kuzu_functions.py:639` |
| `to_minutes(interval)` | Convert to minutes | `kuzu_functions.py:644` |
| `to_seconds(interval)` | Convert to seconds | `kuzu_functions.py:649` |
| `to_milliseconds(interval)` | Convert to milliseconds | `kuzu_functions.py:654` |
| `to_microseconds(interval)` | Convert to microseconds | `kuzu_functions.py:659` |
| `date_part(part, interval)` | Extract interval part | `kuzu_functions.py:664` |

### Array Functions

Array-specific mathematical and similarity functions:

| Function | Description | Implementation |
|----------|-------------|----------------|
| `array_distance(array1, array2)` | Euclidean distance | `kuzu_functions.py:668` |
| `array_inner_product(array1, array2)` | Inner product | `kuzu_functions.py:673` |
| `array_cosine_similarity(array1, array2)` | Cosine similarity | `kuzu_functions.py:678` |
| `array_dot_product(array1, array2)` | Dot product | `kuzu_functions.py:683` |
| `list_distance(list1, list2)` | List distance | `kuzu_functions.py:688` |
| `list_inner_product(list1, list2)` | List inner product | `kuzu_functions.py:693` |
| `list_cosine_similarity(list1, list2)` | List cosine similarity | `kuzu_functions.py:698` |

### Struct Functions

Struct manipulation functions:

| Function | Description | Implementation |
|----------|-------------|----------------|
| `struct_extract(struct, field)` | Extract struct field | `kuzu_functions.py:752` |

### Map Functions

Map manipulation and access functions:

| Function | Description | Implementation |
|----------|-------------|----------------|
| `map(keys, values)` | Create map | `kuzu_functions.py:761` |
| `map_extract(map, key)` | Extract map value | `kuzu_functions.py:766` |
| `element_at(map, key)` | Get element at key | `kuzu_functions.py:771` |
| `cardinality(map)` | Map size | `kuzu_functions.py:776` |
| `map_keys(map)` | Get all keys | `kuzu_functions.py:781` |
| `map_values(map)` | Get all values | `kuzu_functions.py:786` |

### Union Functions

Union type manipulation functions:

| Function | Description | Implementation |
|----------|-------------|----------------|
| `union_value(tag := value)` | Create union | `kuzu_functions.py:795` |
| `union_tag(union)` | Get union tag | `kuzu_functions.py:800` |
| `union_extract(union, tag)` | Extract union value | `kuzu_functions.py:805` |

### Node/Rel Functions

Node and relationship introspection functions:

| Function | Description | Implementation |
|----------|-------------|----------------|
| `ID(node_or_rel)` | Get internal ID | `kuzu_functions.py:814` |
| `LABEL(node_or_rel)` | Get label name | `kuzu_functions.py:819` |
| `LABELS(node_or_rel)` | Get label name (alias) | `kuzu_functions.py:824` |
| `OFFSET(node_or_rel)` | Get ID offset | `kuzu_functions.py:829` |

### Recursive Rel Functions

Recursive path and traversal functions:

| Function | Description | Implementation |
|----------|-------------|----------------|
| `NODES(path)` | Get all nodes from path | `kuzu_query_fields.py:921` |
| `RELS(path)` | Get all relationships from path | `kuzu_query_fields.py:926` |
| `PROPERTIES(nodes_or_rels, property)` | Get property from collection | `kuzu_query_fields.py:931` |
| `IS_TRAIL(path)` | Check if path is trail | `kuzu_query_fields.py:936` |
| `IS_ACYCLIC(path)` | Check if path is acyclic | `kuzu_query_fields.py:941` |
| `LENGTH(path)` | Get path length | `kuzu_query_fields.py:946` |
| `COST(path)` | Get weighted path cost | `kuzu_query_fields.py:951` |

### Blob Functions

Binary data manipulation functions:

| Function | Description | Implementation |
|----------|-------------|----------------|
| `BLOB(data)` | Create blob | `kuzu_functions.py:728` |
| `encode(data, format)` | Encode data | `kuzu_functions.py:733` |
| `decode(blob, format)` | Decode blob | `kuzu_functions.py:738` |
| `octet_length(blob)` | Get blob length | `kuzu_functions.py:743` |

### Hash Functions

Cryptographic hash functions:

| Function | Description | Implementation |
|----------|-------------|----------------|
| `md5(data)` | MD5 hash | `kuzu_functions.py:548` |
| `sha256(data)` | SHA256 hash | `kuzu_functions.py:553` |
| `hash(data)` | Generic hash | `kuzu_functions.py:558` |

### UUID Functions

UUID generation and manipulation functions:

| Function | Description | Implementation |
|----------|-------------|----------------|
| `UUID()` | Parse UUID string | `kuzu_functions.py:567` |
| `gen_random_uuid()` | Generate random UUID | `kuzu_functions.py:572` |

### Utility Functions

Utility and miscellaneous functions:

| Function | Description | Implementation |
|----------|-------------|----------------|
| `coalesce(val1, val2, ...)` | First non-null value | `kuzu_functions.py:445` |
| `ifnull(val1, val2)` | Replace null | `kuzu_functions.py:450` |
| `nullif(val1, val2)` | Return null if equal | `kuzu_functions.py:455` |
| `typeof(value)` | Get value type | `kuzu_functions.py:460` |
| `size(collection)` | Collection size | `kuzu_functions.py:465` |
| `range(start, end, step)` | Generate range | `kuzu_functions.py:470` |
| `greatest(val1, val2, ...)` | Maximum value | `kuzu_functions.py:475` |

### Casting Functions

Type conversion and casting functions:

| Function | Description | Implementation |
|----------|-------------|----------------|
| `CAST(value AS type)` | Cast to type | `kuzu_functions.py:931` |
| `cast(value, type)` | Cast function | `kuzu_functions.py:936` |

### Case Expressions

Conditional expression constructs:

| Function | Description | Implementation |
|----------|-------------|----------------|
| `CASE WHEN ... THEN ... END` | General case | `kuzu_functions.py:943` |
| `CASE value WHEN ... THEN ... END` | Simple case | `kuzu_functions.py:946` |

---

## Operator Reference

KuzuAlchemy implements Kuzu operators with type safety and proper precedence handling.

### Comparison Operators

| Operator | Description | Implementation | Usage |
|----------|-------------|----------------|-------|
| `==` | Equal to | `kuzu_query_expressions.py:45` | `field == value` |
| `!=` | Not equal to | `kuzu_query_expressions.py:50` | `field != value` |
| `<` | Less than | `kuzu_query_expressions.py:55` | `field < value` |
| `<=` | Less than or equal | `kuzu_query_expressions.py:60` | `field <= value` |
| `>` | Greater than | `kuzu_query_expressions.py:65` | `field > value` |
| `>=` | Greater than or equal | `kuzu_query_expressions.py:70` | `field >= value` |

### Logical Operators

| Operator | Description | Implementation | Usage |
|----------|-------------|----------------|-------|
| `AND` | Logical AND | `kuzu_query_expressions.py:75` | `expr1 & expr2` |
| `OR` | Logical OR | `kuzu_query_expressions.py:80` | `expr1 \| expr2` |
| `XOR` | Logical XOR | `kuzu_query_expressions.py:85` | `expr1 ^ expr2` |
| `NOT` | Logical NOT | `kuzu_query_expressions.py:90` | `~expr` |

### Null Operators

| Operator | Description | Implementation | Usage |
|----------|-------------|----------------|-------|
| `IS NULL` | Check if null | `kuzu_query_expressions.py:95` | `field.is_null()` |
| `IS NOT NULL` | Check if not null | `kuzu_query_expressions.py:100` | `field.is_not_null()` |

---

## Model Definition

### Node Models

```python
from kuzualchemy import node, KuzuBaseModel, Field, KuzuDataType
from typing import Optional, List
from datetime import datetime

@node("User")  # Table name in Kuzu
class User(KuzuBaseModel):
    # Primary key
    id: int = Field(kuzu_type=KuzuDataType.INT64, primary_key=True)

    # Basic fields
    name: str = Field(kuzu_type=KuzuDataType.STRING, not_null=True)
    email: Optional[str] = Field(kuzu_type=KuzuDataType.STRING, unique=True, default=None)
    age: int = Field(kuzu_type=KuzuDataType.INT32, default=0)

    # Boolean fields
    is_active: bool = Field(kuzu_type=KuzuDataType.BOOL, default=True)

    # Timestamp fields
    created_at: datetime = Field(
        kuzu_type=KuzuDataType.TIMESTAMP,
        default=KuzuDefaultFunction.CURRENT_TIMESTAMP
    )

    # Array fields
    tags: Optional[List[str]] = Field(
        kuzu_type=ArrayTypeSpecification(element_type=KuzuDataType.STRING),
        default=None
    )
```

### Relationship Models

```python
from kuzualchemy import relationship, KuzuRelationshipBase

@relationship("KNOWS", pairs=[(User, User)])
class Knows(KuzuRelationshipBase):
    since: datetime = Field(kuzu_type=KuzuDataType.TIMESTAMP)
    strength: float = Field(kuzu_type=KuzuDataType.DOUBLE, default=1.0)
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
    def get_foreign_key_fields(cls) -> Dict[str, ForeignKeyMetadata]:
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
field = Field(
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
    foreign_key=foreign_key(
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
    tags: List[str] = Field(
        kuzu_type=ArrayTypeSpecification(element_type=KuzuDataType.STRING),
        default=None
    )
```

### Default Values

```python
from kuzualchemy.constants import KuzuDefaultFunction

class User(KuzuBaseModel):
    # Static defaults
    status: str = Field(kuzu_type=KuzuDataType.STRING, default="active")

    # Function defaults
    created_at: datetime = Field(
        kuzu_type=KuzuDataType.TIMESTAMP,
        default=KuzuDefaultFunction.CURRENT_TIMESTAMP
    )

    # Factory defaults
    uuid_field: str = Field(
        kuzu_type=KuzuDataType.UUID,
        default_factory=lambda: str(uuid.uuid4())
    )
```

---

## Relationships

### Basic Relationships

```python
from kuzualchemy import relationship, KuzuRelationshipBase

@relationship("FOLLOWS", pairs=[(User, User)])
class Follows(KuzuRelationshipBase):
    since: datetime = Field(kuzu_type=KuzuDataType.TIMESTAMP)
    weight: float = Field(kuzu_type=KuzuDataType.DOUBLE, default=1.0)
```

### Multi-Pair Relationships

```python
# Multiple relationship pairs
@relationship("AUTHORED", pairs=[
    (User, Post),
    (User, Comment),
    (Organization, Post)
])
class Authored(KuzuRelationshipBase):
    created_at: datetime = Field(kuzu_type=KuzuDataType.TIMESTAMP)
    role: str = Field(kuzu_type=KuzuDataType.STRING, default="author")
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
    .where(query.fields.name.starts_with("A"))
    .where(query.fields.age.between(20, 40))
    .order_by(query.fields.name.asc())
    .limit(10)
    .all())
```

### Advanced Queries

```python
# Aggregation
query = Query(User, session=session)
count_query = query.group_by(query.fields.age).having(count() > 1)

# Joins (relationships)
query = Query(User, session=session)
joined = query.join(Follows, User.id == Follows.from_node_id)

# Subqueries
subquery = Query(User, session=session).where(query.fields.age > 30)
main_query = Query(Post, session=session).where(
    Post.author_id.in_(subquery.select(User.id))
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
count = query.count()  # int
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
# Manual transactions
session.begin()
try:
    user = User(id=1, name="Alice")
    session.add(user)
    session.commit()
except Exception:
    session.rollback()
    raise

# Context manager
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

@node("Account")
class Account(BaseModel):  # Automatic enum conversion
    status: Status = Field(kuzu_type=KuzuDataType.STRING)

    # BaseModel automatically converts enums to/from string values
```

### Foreign Key Support

```python
from kuzualchemy import foreign_key, CascadeAction

@node("Post")
class Post(KuzuBaseModel):
    id: int = Field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    title: str = Field(kuzu_type=KuzuDataType.STRING)
    author_id: int = Field(
        kuzu_type=KuzuDataType.INT64,
        foreign_key=foreign_key(
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
- `get_foreign_key_fields() -> Dict[str, ForeignKeyMetadata]`: Get foreign key fields

#### KuzuRelationshipBase
Base class for relationship models.

**Methods:**
- Same as KuzuBaseModel plus relationship-specific functionality
- `get_from_node_field() -> str`: Get from node field name
- `get_to_node_field() -> str`: Get to node field name

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

#### Field Function
Field definition with options:

```python
Field(
    default: Any = ...,                                    # Default value
    kuzu_type: Union[KuzuDataType, str, ArrayTypeSpecification], # Kuzu data type
    primary_key: bool = False,                            # Primary key flag
    foreign_key: ForeignKeyMetadata = None,               # Foreign key metadata
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

#### @node() / @kuzu_node()
Mark class as Kuzu node:

```python
@node(
    name: str = None,                                    # Node name (defaults to class name)
    abstract: bool = False,                              # Abstract node flag
    compound_indexes: List[CompoundIndex] = None,        # Compound indexes
    table_constraints: List[str] = None,                 # Table constraints
    properties: Dict[str, Any] = None                    # Additional properties
)
```

#### @relationship() / @kuzu_relationship()
Mark class as Kuzu relationship:

```python
@relationship(
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

This project is licensed under the GPL-3.0 license - see the [LICENSE](LICENSE) file for details.

---

## Conclusion

KuzuAlchemy is an Object-Relational Mapping library for the Kuzu graph database. It provides a SQLAlchemy-like interface for working with graph data.

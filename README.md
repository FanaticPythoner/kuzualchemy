# KuzuAlchemy

KuzuAlchemy is a thin ORM facade over ATP native Kuzu execution.

## Ownership

| Layer | Responsibility |
| --- | --- |
| ATP Rust | DB handles, scheduling, transactions, batching, relationship reads, bulk writes, checkpoint barriers, integrity work. |
| KuzuAlchemy | Decorators, model metadata, expression ASTs, identity map, object materialization. |
| Application code | Store-specific method names and input validation. |

KuzuAlchemy does not own DB locks, page loops, relationship pair planners, schema loops, checkpoint execution, or DB caches.

## Install

```bash
pip install kuzualchemy atp-pipeline
```

## Models

```python
from kuzualchemy import KuzuBaseModel, KuzuDataType, kuzu_field, kuzu_node


@kuzu_node("User")
class User(KuzuBaseModel):
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
```

## Native Session Boundary

```python
from kuzualchemy import KuzuSession, get_all_ddl

session = KuzuSession(db_path="graph.db")
session.connection.schema_apply(get_all_ddl().split(";"))
session.bulk_insert_immediate([User(id=1, name="Ada")])
rows = session.execute("MATCH (u:User) RETURN u.id AS id")
session.close()
```

`get_all_ddl()` emits model metadata DDL. ATP normalizes Kuzu identifiers and executes the schema work through the typed native boundary.

## Queries

```python
from kuzualchemy import Query

rows = Query(User, session=session).filter_by(name="Ada").all()
```

`Query` builds an ORM AST and delegates execution to the session gateway. The gateway submits typed ATP work and returns rows for materialization.

## Bulk Work

```python
session.bulk_update_nodes(User, [{"id": 1, "name": "Grace"}])
session.bulk_delete_nodes(User, [1])
```

Node and relationship routing metadata is extracted from decorators and sent to ATP. ATP performs the execution plan and transaction boundary.

## Checkpoint And Integrity

```python
session.connection.checkpoint()
report = session.connection.snapshot_integrity()
```

The checkpoint call is an ATP barrier ticket. Integrity checks return typed count fields and raise on invalid result shapes.

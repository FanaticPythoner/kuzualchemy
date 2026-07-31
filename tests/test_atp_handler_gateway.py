from __future__ import annotations

from typing import Any

import pytest
from atp_pipeline import DbBulkAction, DbWorkKind
from kuzualchemy.kuzu_connection import KuzuConnection


class _Ticket:
    def __init__(self, payload: Any) -> None:
        self.payload = payload

    def result(self, timeout: float | None = None) -> Any:
        return self.payload


class _GatewayHandler:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any, bool]] = []

    def checkpoint_barrier(self) -> None:
        self.calls.append(("checkpoint_barrier", None, False))

    def execute_kuzu_read(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append(("execute_kuzu_read", (query, parameters or {}), True))
        return [{"n": 1}]

    def submit_work(self, work: Any, *, expect_rows: bool = False, priority: Any = None) -> _Ticket:
        self.calls.append(("submit_work", work, expect_rows))
        return _Ticket({"cypher_results": [[{"n": 1}]]} if expect_rows else {})

    def submit_many_work(
        self,
        works: list[Any],
        *,
        expect_rows: bool = False,
        priority: Any = None,
    ) -> list[_Ticket]:
        self.calls.append(("submit_many_work", works, expect_rows))
        return [_Ticket({}) for _ in works]


class _SingleTableReadHandler:
    def submit_work(self, work: Any, *, expect_rows: bool = False, priority: Any = None) -> _Ticket:
        return _Ticket({"cypher_results": [[{"n": 1}]]})


class _LifecycleHandler:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def flush(self, timeout: float | None) -> None:
        assert timeout is None
        self.calls.append("flush")

    def shutdown(self, timeout: float | None) -> None:
        assert timeout is None
        self.calls.append("shutdown")


def _connection_for_handler(handler: _GatewayHandler) -> KuzuConnection:
    connection = object.__new__(KuzuConnection)
    connection._handler = handler
    connection.db_path = ":memory:"
    connection._closed = False
    return connection


def test_checkpoint_routes_to_handler_barrier() -> None:
    handler = _GatewayHandler()
    connection = _connection_for_handler(handler)

    connection.checkpoint()

    assert handler.calls == [("checkpoint_barrier", None, False)]


def test_read_and_write_batches_route_to_typed_work() -> None:
    handler = _GatewayHandler()
    connection = _connection_for_handler(handler)

    rows = connection.execute_many([("MATCH (n) RETURN n", {})])
    connection.write_many([("CREATE (:N {id: $id})", {"id": 1})])

    assert rows == [[{"n": 1}]]
    read_work = handler.calls[0][1]
    write_work = handler.calls[1][1]
    assert read_work.kind == DbWorkKind.QUERY_READ
    assert write_work.kind == DbWorkKind.QUERY_READ
    assert handler.calls[0][2] is True
    assert handler.calls[1][2] is False


def test_single_read_routes_to_native_atp_read() -> None:
    handler = _GatewayHandler()
    connection = _connection_for_handler(handler)

    rows = connection.execute("MATCH (n) RETURN n", {"limit": 1})

    assert rows == [{"n": 1}]
    assert handler.calls == [
        ("execute_kuzu_read", ("MATCH (n) RETURN n", {"limit": 1}), True)
    ]


def test_close_releases_native_handler_after_ordered_shutdown() -> None:
    handler = _LifecycleHandler()
    connection = object.__new__(KuzuConnection)
    connection.db_path = ":memory:"
    connection._closed = False
    connection._handler = handler

    connection.close()
    connection.close()

    assert handler.calls == ["flush", "shutdown"]
    assert connection._handler is None
    with pytest.raises(RuntimeError, match="closed"):
        connection._open_handler()


def test_bulk_write_nodes_many_routes_to_single_handler_call() -> None:
    handler = _GatewayHandler()
    connection = object.__new__(KuzuConnection)
    connection.db_path = ":memory:"
    connection._closed = False
    connection._handler = handler

    connection.bulk_write_nodes_many(
        [
            (DbBulkAction.CREATE, "A", [{"id": 1}], ["id"]),
            (DbBulkAction.CREATE, "B", [{"id": 2}], ["id"]),
        ]
    )

    assert len(handler.calls) == 1
    kind, works, expect_rows = handler.calls[0]
    assert kind == "submit_many_work"
    assert [work.kind for work in works] == [DbWorkKind.NODE_BULK_WRITE] * 2
    assert [work.node_bulk.label for work in works] == ["A", "B"]
    assert expect_rows is False


def test_bulk_write_nodes_and_relationships_many_preserves_typed_work_items() -> None:
    handler = _GatewayHandler()
    connection = object.__new__(KuzuConnection)
    connection.db_path = ":memory:"
    connection._closed = False
    connection._handler = handler

    connection.bulk_write_nodes_and_relationships_many(
        [(DbBulkAction.CREATE, "A", [{"id": 1}], ["id"])],
        [(DbBulkAction.CREATE, "REL", "A", "B", [{"from_pk": 1, "to_pk": 2}], ["id"], ["id"])],
    )

    assert len(handler.calls) == 1
    kind, works, expect_rows = handler.calls[0]
    assert kind == "submit_many_work"
    assert [work.kind for work in works] == [
        DbWorkKind.NODE_BULK_WRITE,
        DbWorkKind.RELATIONSHIP_BULK_WRITE,
    ]
    assert works[0].node_bulk.label == "A"
    assert works[1].relationship_bulk.rel_type == "REL"
    assert expect_rows is False


def test_kuzu_connection_rejects_batched_read_table_count_mismatch() -> None:
    connection = object.__new__(KuzuConnection)
    connection.db_path = ":memory:"
    connection._closed = False
    connection._handler = _SingleTableReadHandler()

    with pytest.raises(RuntimeError, match="table count"):
        connection.execute_many(
            [
                ("MATCH (a) RETURN a", {}),
                ("MATCH (b) RETURN b", {}),
            ]
        )

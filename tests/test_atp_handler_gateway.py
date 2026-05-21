from __future__ import annotations

from typing import Any

import pytest
from atp_pipeline import DbBulkAction, DbWorkKind
from kuzualchemy.atp_integration import ATPIntegration
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


class _MalformedReadHandler:
    def submit_work(self, work: Any, *, expect_rows: bool = False, priority: Any = None) -> _Ticket:
        return _Ticket({})


class _EmptyReadHandler:
    def submit_work(self, work: Any, *, expect_rows: bool = False, priority: Any = None) -> _Ticket:
        return _Ticket({"cypher_results": []})


class _SingleTableReadHandler:
    def submit_work(self, work: Any, *, expect_rows: bool = False, priority: Any = None) -> _Ticket:
        return _Ticket({"cypher_results": [[{"n": 1}]]})


def _integration_for_handler(handler: _GatewayHandler) -> ATPIntegration:
    integration = object.__new__(ATPIntegration)
    integration._handler = handler
    integration.db_path = ":memory:"
    integration._closed = False
    return integration


def test_checkpoint_routes_to_handler_barrier() -> None:
    handler = _GatewayHandler()
    integration = _integration_for_handler(handler)

    integration.checkpoint()

    assert handler.calls == [("checkpoint_barrier", None, False)]


def test_read_and_write_batches_route_to_typed_work() -> None:
    handler = _GatewayHandler()
    integration = _integration_for_handler(handler)

    rows = integration.execute_many([("MATCH (n) RETURN n", {})])
    integration.write_many([("CREATE (:N {id: $id})", {"id": 1})])

    assert rows == [[{"n": 1}]]
    read_work = handler.calls[0][1]
    write_work = handler.calls[1][1]
    assert read_work.kind == DbWorkKind.QUERY_READ
    assert write_work.kind == DbWorkKind.SCHEMA_APPLY
    assert handler.calls[0][2] is True
    assert handler.calls[1][2] is False


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


def test_kuzu_connection_rejects_malformed_native_read_result() -> None:
    connection = object.__new__(KuzuConnection)
    connection.db_path = ":memory:"
    connection._closed = False
    connection._handler = _MalformedReadHandler()

    with pytest.raises(RuntimeError, match="missing cypher_results"):
        connection.execute("MATCH (n) RETURN n", {})


def test_kuzu_connection_rejects_missing_result_table() -> None:
    connection = object.__new__(KuzuConnection)
    connection.db_path = ":memory:"
    connection._closed = False
    connection._handler = _EmptyReadHandler()

    with pytest.raises(RuntimeError, match="missing first table"):
        connection.execute("MATCH (n) RETURN n", {})


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

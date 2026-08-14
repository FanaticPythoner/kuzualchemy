from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from atp_pipeline import (
    DbBulkAction,
    DbRelationshipEndpointMerge,
    DbRelationshipEndpointMergeSource,
    DbRelationshipEndpointMove,
    DbWorkKind,
)

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
    assert write_work.kind == DbWorkKind.QUERY_WRITE
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
    connection._row_partition_executor = ThreadPoolExecutor(max_workers=1)

    connection.close()
    connection.close()

    assert handler.calls == ["flush", "shutdown"]
    assert connection._handler is None
    assert connection._row_partition_executor is None
    with pytest.raises(RuntimeError, match="closed"):
        connection._open_handler()


def test_row_partition_executor_preserves_order_reuses_pool_and_applies_affinity() -> None:
    connection = object.__new__(KuzuConnection)
    connection.db_path = ":memory:"
    connection._closed = False
    connection._row_partition_executor = ThreadPoolExecutor(max_workers=2)
    executor_identity = id(connection._row_partition_executor)
    inherited_cpu_ids = tuple(sorted(os.sched_getaffinity(0)))
    selected_cpu_ids = inherited_cpu_ids[: min(2, len(inherited_cpu_ids))]

    def capture(value: int) -> tuple[int, int, tuple[int, ...]]:
        return value, threading.get_native_id(), tuple(sorted(os.sched_getaffinity(0)))

    try:
        os.sched_setaffinity(0, selected_cpu_ids)
        first = connection.run_row_partition_tasks(
            [(capture, (value,), {}) for value in range(4)]
        )
        second = connection.run_row_partition_tasks(
            [(capture, (value,), {}) for value in range(4, 8)]
        )
        assert id(connection._row_partition_executor) == executor_identity
    finally:
        os.sched_setaffinity(0, inherited_cpu_ids)
        connection._row_partition_executor.shutdown(wait=True, cancel_futures=False)
        connection._row_partition_executor = None

    assert [row[0] for row in [*first, *second]] == list(range(8))
    assert all(row[2] == selected_cpu_ids for row in [*first, *second])
    assert len({row[1] for row in [*first, *second]}) <= 2


def test_row_partition_executor_scope_uses_caller_pool_and_restores_owner_pool() -> None:
    connection = object.__new__(KuzuConnection)
    connection.db_path = ":memory:"
    connection._closed = False
    connection._row_partition_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="owned-row",
    )
    shared_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="phase-row",
    )

    def capture() -> str:
        return threading.current_thread().name

    try:
        with connection.row_partition_executor_scope(shared_executor):
            shared_result = connection.run_row_partition_tasks([(capture, (), {})])
        owned_result = connection.run_row_partition_tasks([(capture, (), {})])
    finally:
        shared_executor.shutdown(wait=True, cancel_futures=False)
        connection._row_partition_executor.shutdown(wait=True, cancel_futures=False)
        connection._row_partition_executor = None

    assert shared_result[0].startswith("phase-row")
    assert owned_result[0].startswith("owned-row")


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


def test_bulk_write_nodes_and_relationships_many_orders_endpoint_dependent_phases() -> None:
    handler = _GatewayHandler()
    connection = object.__new__(KuzuConnection)
    connection.db_path = ":memory:"
    connection._closed = False
    connection._handler = handler

    connection.bulk_write_nodes_and_relationships_many(
        [(DbBulkAction.CREATE, "A", [{"id": 1}], ["id"])],
        [(DbBulkAction.CREATE, "REL", "A", "B", [{"from_pk": 1, "to_pk": 2}], ["id"], ["id"])],
    )

    assert len(handler.calls) == 2
    node_kind, node_works, node_expect_rows = handler.calls[0]
    relationship_kind, relationship_works, relationship_expect_rows = handler.calls[1]
    assert node_kind == relationship_kind == "submit_many_work"
    assert [work.kind for work in node_works] == [DbWorkKind.NODE_BULK_WRITE]
    assert [work.kind for work in relationship_works] == [
        DbWorkKind.RELATIONSHIP_BULK_WRITE
    ]
    assert node_works[0].node_bulk.label == "A"
    assert relationship_works[0].relationship_bulk.rel_type == "REL"
    assert node_expect_rows is relationship_expect_rows is False


def test_endpoint_canonicalization_routes_moves_and_merges_to_one_work() -> None:
    handler = _GatewayHandler()
    connection = _connection_for_handler(handler)
    endpoint_move = DbRelationshipEndpointMove(
        rel_type="REL",
        old_from_label="A",
        old_to_label="B",
        new_from_label="A",
        new_to_label="C",
        old_from_key_field="id",
        old_to_key_field="id",
        new_from_key_field="id",
        new_to_key_field="id",
        identity_fields=["site"],
        property_fields=["active"],
        rows=[
            {
                "old_from_pk": 1,
                "old_to_pk": 2,
                "new_from_pk": 1,
                "new_to_pk": 3,
                "site": 10,
                "active": True,
            }
        ],
    )
    endpoint_merge = DbRelationshipEndpointMerge(
        rel_type="REL",
        new_from_label="A",
        new_to_label="C",
        new_from_key_field="id",
        new_to_key_field="id",
        identity_fields=["site"],
        property_fields=["active"],
        sources=[
            DbRelationshipEndpointMergeSource(
                old_from_label="A",
                old_to_label="B",
                old_from_key_field="id",
                old_to_key_field="id",
                rows=[{"old_from_pk": 1, "old_to_pk": 2, "site": 20}],
            )
        ],
        target_row={
            "new_from_pk": 1,
            "new_to_pk": 3,
            "site": 20,
            "active": True,
        },
        target_preexisting=True,
    )

    connection.canonicalize_relationship_endpoints([endpoint_move], [endpoint_merge])

    assert len(handler.calls) == 1
    kind, work, expect_rows = handler.calls[0]
    assert kind == "submit_work"
    assert work.kind == DbWorkKind.RELATIONSHIP_ENDPOINT_CANONICALIZE
    assert work.relationship_endpoint_moves == [endpoint_move]
    assert work.relationship_endpoint_merges == [endpoint_merge]
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

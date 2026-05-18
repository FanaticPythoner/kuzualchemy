from __future__ import annotations

from typing import Any

from atp_pipeline import DbWorkKind
from kuzualchemy.atp_integration import ATPIntegration


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


def _integration_for_handler(handler: _GatewayHandler) -> ATPIntegration:
    integration = object.__new__(ATPIntegration)
    integration._handler = handler
    integration.db_path = ":memory:"
    integration._closed = False
    return integration


def test_checkpoint_routes_to_handler_barrier() -> None:
    handler = _GatewayHandler()
    integration = _integration_for_handler(handler)

    integration.checkpoint_barrier()

    assert handler.calls == [("checkpoint_barrier", None, False)]


def test_read_and_write_batches_route_to_typed_work() -> None:
    handler = _GatewayHandler()
    integration = _integration_for_handler(handler)

    rows = integration.read_many([("MATCH (n) RETURN n", {})])
    integration.write_many([("CREATE (:N {id: $id})", {"id": 1})])

    assert rows == [[{"n": 1}]]
    read_work = handler.calls[0][1]
    write_work = handler.calls[1][1]
    assert read_work.kind == DbWorkKind.QUERY_READ
    assert write_work.kind == DbWorkKind.SCHEMA_APPLY
    assert handler.calls[0][2] is True
    assert handler.calls[1][2] is False

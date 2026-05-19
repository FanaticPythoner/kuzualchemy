from __future__ import annotations
from pathlib import Path
from typing import Any, Iterable
from atp_pipeline import (
    ATPHandler,
    DatabaseType,
    DbBulkAction,
    DbBulkEntity,
    DbBulkRelationship,
    DbStatement,
    DbWorkKind,
    DbWorkSpec,
    OpPriority,
)
from .constants import ErrorMessages
from .kuzu_relationship_read import relationship_read_work
class KuzuConnection:
    """Submit Kuzu work through the native ATP handler."""
    def __init__(self, db_path: str | Path) -> None:
        raw_path = str(db_path)
        self.db_path = raw_path if raw_path == ":memory:" else str(Path(raw_path).resolve())
        self._handler = ATPHandler(DatabaseType.KUZU, {"db_path": self.db_path})
        self._closed = False
    def _open_handler(self) -> ATPHandler:
        if getattr(self, "_closed", False):
            raise RuntimeError(ErrorMessages.CONNECTION_CLOSED)
        return self._handler
    def execute(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        result = self._submit_work(
            DbWorkSpec(DbWorkKind.QUERY_READ, statements=[_statement(query, parameters or {})]),
            expect_rows=True,
        )
        return _first_table(result)
    def execute_write(self, query: str, parameters: dict[str, Any] | None = None) -> None:
        self._submit_work(
            DbWorkSpec(DbWorkKind.SCHEMA_APPLY, statements=[_statement(query, parameters or {})]),
            expect_rows=False,
        )
    def execute_many(self, queries: Iterable[tuple[str, dict[str, Any]]]) -> list[list[dict[str, Any]]]:
        statements = [_statement(query, params) for query, params in queries]
        if not statements:
            return []
        result = self._submit_work(
            DbWorkSpec(DbWorkKind.QUERY_READ, statements=statements),
            expect_rows=True,
        )
        tables = _tables(result)
        if len(tables) != len(statements):
            raise RuntimeError("native DB result table count must match statement count")
        return tables
    def write_many(self, queries: Iterable[tuple[str, dict[str, Any]]]) -> None:
        statements = [_statement(query, params) for query, params in queries]
        if statements:
            self._submit_work(
                DbWorkSpec(DbWorkKind.SCHEMA_APPLY, statements=statements),
                expect_rows=False,
            )
    def read_relationships(
        self,
        *,
        relationship: str,
        alias: str,
        direction: str,
        pairs: list[dict[str, str]],
        pairs_subset: list[int],
    ) -> list[dict[str, Any]]:
        result = self._submit_work(
            relationship_read_work(
                relationship=relationship,
                alias=alias,
                direction=direction,
                pairs=pairs,
                pairs_subset=pairs_subset,
            ),
            expect_rows=True,
        )
        return _first_table(result)
    def schema_apply(self, statements: Iterable[str]) -> None:
        payload = [_statement(statement, {}) for statement in statements if statement.strip()]
        if payload:
            self._submit_work(
                DbWorkSpec(DbWorkKind.SCHEMA_APPLY, statements=payload),
                expect_rows=False,
            )
    def snapshot_integrity(self) -> dict[str, Any]:
        work = DbWorkSpec(
            DbWorkKind.SNAPSHOT_INTEGRITY,
            statements=[
                DbStatement("MATCH (n) RETURN COUNT(n) as count", {}),
                DbStatement("MATCH ()-[r]->() RETURN COUNT(r) as count", {}),
                DbStatement("CALL SHOW_TABLES() RETURN *", {}),
            ],
        )
        result = self._submit_work(
            work,
            expect_rows=True,
            priority=OpPriority.HIGH,
        )
        tables = result.get("cypher_results") if isinstance(result, dict) else None
        if not isinstance(tables, list) or len(tables) != 3:
            raise RuntimeError("snapshot integrity result must contain three tables")
        node_rows, relationship_rows, table_rows = tables
        return {
            "node_count": _count_table(node_rows, "node_count"),
            "relationship_count": _count_table(relationship_rows, "relationship_count"),
            "table_count": len(table_rows),
        }
    def checkpoint(self) -> None:
        self._open_handler().checkpoint_barrier()
    def bulk_write_nodes(
        self,
        action: DbBulkAction,
        label: str,
        rows: list[dict[str, Any]],
        key_fields: list[str],
    ) -> None:
        work = DbWorkSpec(
            DbWorkKind.NODE_BULK_WRITE,
            node_bulk=DbBulkEntity(action, label, rows, key_fields),
        )
        self._submit_work(work, expect_rows=False)
    def bulk_write_relationships(
        self,
        action: DbBulkAction,
        rel_type: str,
        from_label: str,
        to_label: str,
        rows: list[dict[str, Any]],
        from_key_fields: list[str],
        to_key_fields: list[str],
    ) -> None:
        work = DbWorkSpec(
            DbWorkKind.RELATIONSHIP_BULK_WRITE,
            relationship_bulk=DbBulkRelationship(
                action,
                rel_type,
                from_label,
                to_label,
                rows,
                from_key_fields,
                to_key_fields,
            ),
        )
        self._submit_work(work, expect_rows=False)
    def close(self) -> None:
        if self._closed:
            return
        self._handler.flush(None)
        self._handler.shutdown(None)
        self._closed = True
    def _submit_work(
        self,
        work: DbWorkSpec,
        *,
        expect_rows: bool,
        priority: OpPriority = OpPriority.NORMAL,
    ) -> Any:
        return self._open_handler().submit_work(
            work,
            expect_rows=expect_rows,
            priority=priority,
        ).result(None)
def _statement(query: str, parameters: dict[str, Any]) -> DbStatement:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(parameters, dict):
        raise TypeError("parameters must be a dict")
    return DbStatement(query, parameters)
def _tables(result: Any) -> list[list[dict[str, Any]]]:
    if not isinstance(result, dict):
        raise RuntimeError("native DB result must be a dictionary")
    tables = result.get("cypher_results")
    if not isinstance(tables, list):
        raise RuntimeError("native DB result missing cypher_results table list")
    for table in tables:
        if not isinstance(table, list):
            raise RuntimeError("native DB result table must be a list")
        for row in table:
            if not isinstance(row, dict):
                raise RuntimeError("native DB result row must be a dictionary")
    return tables
def _first_table(result: Any) -> list[dict[str, Any]]:
    tables = _tables(result)
    if not tables:
        raise RuntimeError("native DB result missing first table")
    return tables[0]
def _count_table(rows: Any, name: str) -> int:
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{name} missing count row")
    first = rows[0]
    if not isinstance(first, dict) or "count" not in first:
        raise RuntimeError(f"{name} missing count field")
    value = first["count"]
    if not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{name} count must be a non-negative integer")
    return value

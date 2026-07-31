from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import atp_pipeline as atp
from atp_pipeline import (
    ATPHandler,
    DatabaseType,
    DbBulkAction,
    DbBulkMergePolicy,
    DbRelationshipEndpointReplace,
    DbStatement,
)

from .constants import ErrorMessages


class KuzuConnection:
    """Submit Kuzu work through ATP."""

    def __init__(self, db_path: str | Path) -> None:
        raw_path = str(db_path)
        self.db_path = raw_path if raw_path == ":memory:" else str(Path(raw_path).resolve())
        self._handler: ATPHandler | None = ATPHandler(
            DatabaseType.KUZU,
            {"db_path": self.db_path},
        )
        self._closed = False

    def _open_handler(self) -> ATPHandler:
        handler = getattr(self, "_handler", None)
        if getattr(self, "_closed", False) or handler is None:
            raise RuntimeError(ErrorMessages.CONNECTION_CLOSED)
        return handler

    def execute(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if atp.is_schema_or_admin_query(query) or atp.is_write_query(query):
            self.execute_write(query, parameters)
            return []
        return atp.execute_kuzu(self._open_handler(), query, parameters or {})

    def execute_write(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        atp.execute_kuzu_write(self._open_handler(), query, parameters or {})

    def execute_many(
        self,
        queries: Iterable[tuple[str, dict[str, Any]]],
    ) -> list[list[dict[str, Any]]]:
        return atp.execute_kuzu_many(self._open_handler(), queries, mode="read")

    def write_many(self, queries: Iterable[tuple[str, dict[str, Any]]]) -> None:
        atp.execute_kuzu_many(self._open_handler(), queries, mode="write")

    def execute_write_many_returning(
        self,
        queries: Iterable[tuple[str, dict[str, Any]]],
    ) -> list[list[dict[str, Any]]]:
        return atp.execute_kuzu_write_many_returning(self._open_handler(), queries)

    def iterate(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        *,
        page_size: int = 1000,
        prefetch_pages: int = 1,
    ):
        return atp.iter_kuzu(
            self._open_handler(),
            query,
            parameters or {},
            page_size=page_size,
            prefetch_pages=prefetch_pages,
        )

    def read_pages(
        self,
        query: str,
        parameters: dict[str, Any] | None,
        offsets: list[int],
        limit: int,
    ) -> list[list[dict[str, Any]]]:
        return atp.read_kuzu_pages(self._open_handler(), query, parameters or {}, offsets, limit)

    def read_relationships(
        self,
        *,
        relationship: str,
        alias: str,
        direction: str,
        pairs: list[dict[str, str]],
        pairs_subset: list[int],
        filters: list[DbStatement] | None = None,
        page_size: int = 0,
    ) -> list[dict[str, Any]]:
        return atp.read_kuzu_relationships(
            self._open_handler(),
            relationship=relationship,
            alias=alias,
            direction=direction,
            pairs=pairs,
            pairs_subset=pairs_subset,
            filters=filters,
            page_size=page_size,
        )

    def schema_apply(self, statements: Iterable[str]) -> None:
        atp.execute_kuzu_many(
            self._open_handler(),
            [(statement, {}) for statement in statements if statement.strip()],
            mode="schema",
        )

    def find_node_labels_for_primary_key(
        self,
        node_specs: Iterable[tuple[str, str]],
        primary_key_value: Any,
    ) -> list[str]:
        return atp.find_kuzu_node_labels_for_primary_key(
            self._open_handler(),
            node_specs,
            primary_key_value,
        )

    def snapshot_integrity(self) -> dict[str, Any]:
        return atp.kuzu_snapshot_integrity(self._open_handler())

    def checkpoint(self) -> None:
        self._open_handler().checkpoint_barrier()

    def capability_report(self) -> Any:
        return self._open_handler().get_capability_report()

    def metrics_snapshot(self) -> str:
        return self._open_handler().metrics_snapshot()

    def bulk_write_nodes(
        self,
        action: DbBulkAction,
        label: str,
        rows: list[dict[str, Any]],
        key_fields: list[str],
        merge_policies: dict[str, DbBulkMergePolicy | str] | None = None,
    ) -> None:
        atp.bulk_write_kuzu_nodes(
            self._open_handler(),
            action,
            label,
            rows,
            key_fields,
            merge_policies,
        )

    def bulk_write_nodes_many(self, batches: Iterable[tuple[Any, ...]]) -> None:
        atp.bulk_write_kuzu_nodes_many(self._open_handler(), batches)

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
        atp.bulk_write_kuzu_relationships(
            self._open_handler(),
            action,
            rel_type,
            from_label,
            to_label,
            rows,
            from_key_fields,
            to_key_fields,
        )

    def bulk_write_relationships_many(
        self,
        batches: Iterable[
            tuple[DbBulkAction, str, str, str, list[dict[str, Any]], list[str], list[str]]
        ],
    ) -> None:
        atp.bulk_write_kuzu_relationships_many(self._open_handler(), batches)

    def replace_relationship_endpoints(
        self,
        replacements: Iterable[DbRelationshipEndpointReplace],
    ) -> None:
        atp.replace_kuzu_relationship_endpoints(self._open_handler(), replacements)

    def bulk_write_nodes_and_relationships_many(
        self,
        node_batches: Iterable[tuple[Any, ...]],
        relationship_batches: Iterable[
            tuple[DbBulkAction, str, str, str, list[dict[str, Any]], list[str], list[str]]
        ],
    ) -> None:
        atp.bulk_write_kuzu_nodes_and_relationships_many(
            self._open_handler(),
            node_batches,
            relationship_batches,
        )

    def create_generated_nodes(
        self,
        specs: Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return atp.create_kuzu_generated_nodes(self._open_handler(), specs)

    def create_generated_relationships(
        self,
        specs: Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return atp.create_kuzu_generated_relationships(self._open_handler(), specs)

    def close(self) -> None:
        if self._closed:
            return
        handler = self._open_handler()
        handler.flush(None)
        handler.shutdown(None)
        self._handler = None
        self._closed = True

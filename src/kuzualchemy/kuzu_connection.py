from __future__ import annotations

import os
import threading
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import atp_pipeline as atp
from atp_pipeline import (
    ATPHandler,
    DatabaseType,
    DbBulkAction,
    DbBulkMergePolicy,
    DbRelationshipEndpointMerge,
    DbRelationshipEndpointMove,
    DbStatement,
)

from .constants import ErrorMessages

RowPartitionTask = tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]
_ROW_PARTITION_THREAD_STATE = threading.local()


def _process_cpu_capacity() -> int:
    capacity = os.process_cpu_count()
    if capacity is None or capacity < 1:
        raise RuntimeError("process CPU count is unavailable")
    return capacity


def _current_cpu_affinity() -> tuple[int, ...] | None:
    affinity_reader = getattr(os, "sched_getaffinity", None)
    if affinity_reader is None:
        return None
    try:
        cpu_ids = tuple(sorted(affinity_reader(0)))
    except OSError as exc:
        raise RuntimeError(f"row partition CPU-affinity discovery failed: {exc}") from exc
    if not cpu_ids:
        raise RuntimeError("row partition CPU affinity is empty")
    return cpu_ids


def _execute_row_partition_task(
    cpu_ids: tuple[int, ...] | None,
    function: Callable[..., Any],
    arguments: tuple[Any, ...],
    keyword_arguments: dict[str, Any],
) -> Any:
    if cpu_ids is not None and getattr(_ROW_PARTITION_THREAD_STATE, "cpu_ids", None) != cpu_ids:
        affinity_writer = getattr(os, "sched_setaffinity", None)
        affinity_reader = getattr(os, "sched_getaffinity", None)
        if affinity_writer is None or affinity_reader is None:
            raise RuntimeError("row partition CPU-affinity activation is unavailable")
        try:
            affinity_writer(0, cpu_ids)
            observed = tuple(sorted(affinity_reader(0)))
        except OSError as exc:
            raise RuntimeError(f"row partition CPU-affinity activation failed: {exc}") from exc
        if observed != cpu_ids:
            raise RuntimeError(
                f"row partition CPU affinity mismatch: observed={observed}, expected={cpu_ids}"
            )
        _ROW_PARTITION_THREAD_STATE.cpu_ids = cpu_ids
    return function(*arguments, **keyword_arguments)


class KuzuConnection:
    """Submit Kuzu work through ATP."""

    def __init__(self, db_path: str | Path) -> None:
        raw_path = str(db_path)
        self.db_path = raw_path if raw_path == ":memory:" else str(Path(raw_path).resolve())
        self._handler: ATPHandler | None = ATPHandler(
            DatabaseType.KUZU,
            {"db_path": self.db_path},
        )
        self._row_partition_executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=_process_cpu_capacity(),
            thread_name_prefix="kuzualchemy-row",
        )
        self._row_partition_executor_override: ContextVar[Executor | None] = ContextVar(
            "kuzualchemy_row_partition_executor",
            default=None,
        )
        self._closed = False

    def _open_handler(self) -> ATPHandler:
        handler = getattr(self, "_handler", None)
        if getattr(self, "_closed", False) or handler is None:
            raise RuntimeError(ErrorMessages.CONNECTION_CLOSED)
        return handler

    def _open_row_partition_executor(self) -> Executor:
        override = getattr(self, "_row_partition_executor_override", None)
        if override is not None:
            selected = override.get()
            if selected is not None:
                if getattr(self, "_closed", False):
                    raise RuntimeError(ErrorMessages.CONNECTION_CLOSED)
                return selected
        executor = getattr(self, "_row_partition_executor", None)
        if getattr(self, "_closed", False) or executor is None:
            raise RuntimeError(ErrorMessages.CONNECTION_CLOSED)
        return executor

    @contextmanager
    def row_partition_executor_scope(self, executor: Executor) -> Iterator[None]:
        if not isinstance(executor, Executor):
            raise TypeError("row partition executor must implement concurrent.futures.Executor")
        if getattr(self, "_closed", False):
            raise RuntimeError(ErrorMessages.CONNECTION_CLOSED)
        override = getattr(self, "_row_partition_executor_override", None)
        if override is None:
            override = ContextVar(
                "kuzualchemy_row_partition_executor",
                default=None,
            )
            self._row_partition_executor_override = override
        token = override.set(executor)
        try:
            yield
        finally:
            override.reset(token)

    def run_row_partition_tasks(
        self,
        tasks: Iterable[RowPartitionTask],
    ) -> list[Any]:
        task_list = list(tasks)
        if not task_list:
            return []
        executor = self._open_row_partition_executor()
        cpu_ids = _current_cpu_affinity()
        futures = [
            executor.submit(
                _execute_row_partition_task,
                cpu_ids,
                function,
                arguments,
                keyword_arguments,
            )
            for function, arguments, keyword_arguments in task_list
        ]
        results: list[Any] = []
        first_error: BaseException | None = None
        for future in futures:
            try:
                results.append(future.result())
            except BaseException as exc:
                results.append(None)
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error.with_traceback(first_error.__traceback__)
        return results

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
        total_rows: int | None = None,
    ):
        return atp.iter_kuzu(
            self._open_handler(),
            query,
            parameters or {},
            page_size=page_size,
            prefetch_pages=prefetch_pages,
            total_rows=total_rows,
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
    ) -> list[dict[str, Any]]:
        return atp.read_kuzu_relationships(
            self._open_handler(),
            relationship=relationship,
            alias=alias,
            direction=direction,
            pairs=pairs,
            pairs_subset=pairs_subset,
            filters=filters,
        )

    def iterate_relationships(
        self,
        *,
        relationship: str,
        alias: str,
        direction: str,
        pairs: list[dict[str, str]],
        pairs_subset: list[int],
        filters: list[DbStatement] | None,
        page_size: int,
        prefetch_pages: int,
    ) -> Iterator[dict[str, Any]]:
        return atp.iter_kuzu_relationships(
            self._open_handler(),
            relationship=relationship,
            alias=alias,
            direction=direction,
            pairs=pairs,
            pairs_subset=pairs_subset,
            filters=filters,
            page_size=page_size,
            prefetch_pages=prefetch_pages,
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

    def canonicalize_relationship_endpoints(
        self,
        moves: Iterable[DbRelationshipEndpointMove],
        merges: Iterable[DbRelationshipEndpointMerge],
    ) -> None:
        atp.canonicalize_kuzu_relationship_endpoints(
            self._open_handler(),
            moves,
            merges,
        )

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
        executor = getattr(self, "_row_partition_executor", None)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
            self._row_partition_executor = None
        handler = self._open_handler()
        handler.flush(None)
        handler.shutdown(None)
        self._handler = None
        self._closed = True

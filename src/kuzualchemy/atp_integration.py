from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import threading
import atexit
import logging
import time
import re

from atp_pipeline import (
    ATPHandler,
    DatabaseType,
    OperationSpec,
    ReturnSpec,
    OpKind,
    ReturnMode,
)


_HANDLER_LOCK = threading.RLock()
_HANDLERS: Dict[str, Union[ATPHandler, None]] = {}
_HANDLER_REFCOUNTS: Dict[str, int] = {}
_KUZU_INITIALIZED: set[str] = set()  # Track initialized db_paths to avoid repeated calls
logger = logging.getLogger(__name__)


def _shutdown_all_handlers() -> None:
    # Best-effort shutdown at interpreter exit.
    handlers: list[tuple[str, ATPHandler]] = []
    with _HANDLER_LOCK:
        if _HANDLERS:
            handlers = list(_HANDLERS.items())
        _HANDLERS.clear()
        _HANDLER_REFCOUNTS.clear()
        _KUZU_INITIALIZED.clear()

    for key, handler in handlers:
        flush_err: Exception | None = None
        shutdown_err: Exception | None = None
        t0 = time.perf_counter()
        try:
            handler.flush(None)
        except Exception as exc:  # pylint: disable=broad-except
            flush_err = exc
            logger.error("kuzualchemy.atp.handler.shutdown_all.flush_failed db_path=%s error=%s", key, exc)
        t1 = time.perf_counter()
        try:
            handler.shutdown(None)
        except Exception as exc:  # pylint: disable=broad-except
            shutdown_err = exc
            logger.error("kuzualchemy.atp.handler.shutdown_all.shutdown_failed db_path=%s error=%s", key, exc)
        t2 = time.perf_counter()
        logger.info(
            "kuzualchemy.atp.handler.shutdown_all.done db_path=%s flush_seconds=%.6f shutdown_seconds=%.6f total_seconds=%.6f",
            key,
            t1 - t0,
            t2 - t1,
            t2 - t0,
        )
        if flush_err or shutdown_err:
            continue


atexit.register(_shutdown_all_handlers)


def _acquire_handler(db_path: str) -> ATPHandler:
    # ':memory:' is treated as ephemeral per-instance. It should not be cached across sessions,
    # otherwise separate in-memory graphs would alias and tests would interfere.
    if db_path == ":memory:":
        cfg: Dict[str, Any] = {"db_path": db_path}
        handler: ATPHandler = ATPHandler(DatabaseType.KUZU, cfg)
        _ = handler.get_capability_report()
        logger.debug("kuzualchemy.atp.handler.acquire.ephemeral db_path=%s", db_path)
        return handler

    key = str(Path(db_path).resolve())
    with _HANDLER_LOCK:
        handler = _HANDLERS.get(key)
        if handler is not None:
            _HANDLER_REFCOUNTS[key] = _HANDLER_REFCOUNTS.get(key, 0) + 1
            logger.debug(
                "kuzualchemy.atp.handler.acquire.reuse db_path=%s refcount=%d",
                key,
                _HANDLER_REFCOUNTS[key],
            )
            return handler

        # Create new handler and store for process lifetime
        cfg: Dict[str, Any] = {"db_path": key}
        handler: ATPHandler = ATPHandler(DatabaseType.KUZU, cfg)
        # Force capability negotiation early for explicit failure
        _ = handler.get_capability_report()
        _HANDLERS[key] = handler
        _HANDLER_REFCOUNTS[key] = _HANDLER_REFCOUNTS.get(key, 0) + 1
        logger.debug("kuzualchemy.atp.handler.acquire.create db_path=%s refcount=%d", key, _HANDLER_REFCOUNTS[key])
        return handler


def _release_handler(db_path: str, timeout: Optional[float]) -> None:
    # ':memory:' is ephemeral and is handled by the instance.
    if db_path == ":memory:":
        return

    # For file-backed DBs, release when no sessions remain (refcount reaches zero).
    key = str(Path(db_path).resolve())
    handler: ATPHandler | None = None
    do_flush = False
    do_shutdown = False
    new_refcount: int | None = None
    with _HANDLER_LOCK:
        handler = _HANDLERS.get(key)
        if handler is None:
            return
        refcount = _HANDLER_REFCOUNTS.get(key, 0)
        if refcount <= 1:
            _HANDLERS.pop(key, None)
            _HANDLER_REFCOUNTS.pop(key, None)
            _KUZU_INITIALIZED.discard(key)
            do_flush = True
            do_shutdown = True
            new_refcount = 0
        else:
            _HANDLER_REFCOUNTS[key] = refcount - 1
            do_flush = True
            new_refcount = _HANDLER_REFCOUNTS[key]

    logger.debug(
        "kuzualchemy.atp.handler.release db_path=%s refcount=%s shutdown=%s",
        key,
        new_refcount,
        do_shutdown,
    )

    if handler is None or not do_flush:
        return

    flush_err: Exception | None = None
    shutdown_err: Exception | None = None
    t0 = time.perf_counter()
    try:
        handler.flush(timeout)
    except Exception as exc:  # pylint: disable=broad-except
        flush_err = exc
        logger.error("kuzualchemy.atp.handler.release.flush_failed db_path=%s error=%s", key, exc)
    t1 = time.perf_counter()

    if do_shutdown:
        try:
            handler.shutdown(timeout)
        except Exception as exc:  # pylint: disable=broad-except
            shutdown_err = exc
            logger.error("kuzualchemy.atp.handler.release.shutdown_failed db_path=%s error=%s", key, exc)
        t2 = time.perf_counter()
        logger.debug(
            "kuzualchemy.atp.handler.release.done db_path=%s flush_seconds=%.6f shutdown_seconds=%.6f total_seconds=%.6f",
            key,
            t1 - t0,
            t2 - t1,
            t2 - t0,
        )
    else:
        logger.debug(
            "kuzualchemy.atp.handler.release.flushed db_path=%s flush_seconds=%.6f",
            key,
            t1 - t0,
        )

    if flush_err is not None:
        raise flush_err
    if shutdown_err is not None:
        raise shutdown_err


class ATPIntegration:
    """Integration with ATP pipeline for Kuzu database operations."""

    def __init__(self, db_path: str | Path) -> None:
        raw = str(db_path)
        if raw == ":memory:":
            self._db_path = ":memory:"
            self._ephemeral_handler: Optional[ATPHandler] = _acquire_handler(":memory:")
            self._handler = self._ephemeral_handler
        else:
            self._db_path = str(Path(db_path).resolve())
            self._ephemeral_handler = None
            # Acquire or create a shared handler per db_path
            self._handler = _acquire_handler(self._db_path)

    def run_cypher(self, query: str, params: Optional[Dict[str, Any]], expect_rows: bool) -> List[Dict[str, Any]]:
        # FAST PATH: For read-only queries expecting rows, bypass ATP pipeline entirely
        # and use the readonly connection pool directly for ~10x faster execution.
        if expect_rows and self._is_readonly_query(query):
            return self._run_cypher_fast(query, params)
        
        # SLOW PATH: Full ATP pipeline for write queries or non-row-returning queries
        returns = ReturnSpec(modes=ReturnMode.POST_CYPHER if expect_rows else ReturnMode.NONE)
        spec = OperationSpec(
            kind=OpKind.RUN_CYPHER,
            cypher=[query],
            props=params or {},
            returns=returns,
            context={"db_path": self._db_path},
        )
        tk = self._handler.submit(spec)
        try:
            res = tk.result(None)
        except Exception as e:
            # Normalize exceptions to RuntimeError for consistent handling by callers/tests
            raise RuntimeError(f"Query error: {e}") from e
        if expect_rows:
            cr = res.get("cypher_results")
            if isinstance(cr, list) and len(cr) > 0 and isinstance(cr[0], list):
                first = cr[0]
                if isinstance(first, list):
                    return first  # type: ignore[return-value]
                return []
        return []

    def _is_readonly_query(self, query: str) -> bool:
        """Check if a query is read-only (no write clauses).
        
        Read-only queries can use the fast path (readonly connection pool)
        instead of the full ATP transactional pipeline.
        """
        q = re.sub(r"\s+", " ", query.strip()).upper()
        # Write clause keywords that indicate a non-readonly query
        write_keywords = (
            'CREATE', 'DELETE', 'DETACH', 'SET', 'REMOVE', 'MERGE',
            'DROP', 'ALTER', 'COPY', 'CALL', 'BEGIN', 'COMMIT', 'ROLLBACK'
        )
        for kw in write_keywords:
            # Check for keyword as whole word (not substring)
            if f' {kw} ' in f' {q} ' or q.startswith(f'{kw} ') or f' {kw}(' in f' {q} ':
                return False
        return True

    def _run_cypher_fast(self, query: str, params: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute a read-only query using the readonly connection pool directly.
        
        This bypasses the full ATP transactional pipeline for ~10x faster execution
        on read queries. Uses execute_parallel_queries with a single query.
        """
        from atp_pipeline import execute_parallel_queries, ensure_kuzu_initialized
        
        try:
            # Ensure database is initialized only once per db_path
            if self._db_path not in _KUZU_INITIALIZED:
                ensure_kuzu_initialized(self._db_path)
                _KUZU_INITIALIZED.add(self._db_path)
            results = execute_parallel_queries(self._db_path, [(query, params or {})])
            if results and len(results) > 0:
                return results[0]
            return []
        except Exception as e:
            raise RuntimeError(f"Query error: {e}") from e

    def run_cypher_parallel(self, queries: List[tuple[str, Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        """Execute multiple Cypher queries in parallel using the readonly connection pool.
        
        This leverages the Rust-side parallel execution with rayon and the readonly
        connection pool for GIL-independent parallel query execution.
        
        Args:
            queries: List of (query_string, params_dict) tuples to execute in parallel
            
        Returns:
            List of result lists (one per query, in same order as input)
        """
        if not queries:
            return []
        
        # Import here to avoid circular imports
        from atp_pipeline import execute_parallel_queries
        
        # Convert to format expected by Rust: list of (query, params) tuples
        query_tuples = [(q, p or {}) for q, p in queries]
        
        try:
            results = execute_parallel_queries(self._db_path, query_tuples)
            return results
        except Exception as e:
            raise RuntimeError(f"Parallel query error: {e}") from e

    def create_node(self, label: str, props: Optional[Dict[str, Any]] = None, pk: Optional[Dict[str, Any]] = None, return_object: bool = False) -> Optional[Dict[str, Any]]:
        returns = None
        # Keep returns None to avoid extra overhead unless object explicitly requested
        if return_object:
            returns = ReturnSpec(modes=ReturnMode.OBJECT)
        spec = OperationSpec(
            kind=OpKind.CREATE_NODE,
            label=label,
            pk=pk or {},
            props=props or {},
            returns=returns,
            context={"db_path": self._db_path, "label": label, "pk": (pk or {})},
        )
        tk = self._handler.submit(spec)
        try:
            res = tk.result(None)
        except Exception as e:
            raise RuntimeError(f"Execute failed: {e}") from e
        if return_object and isinstance(res, dict):
            obj = res.get("object")
            if isinstance(obj, dict):
                return obj
        return None

    def create_nodes(self, label: str, rows: List[Dict[str, Any]], *, return_rows: bool = False, pk_fields: Optional[List[str]] = None) -> Optional[List[Dict[str, Any]]]:
        # Bulk create via rows list (UNWIND in executor). When return_rows=True, request cypher_results rows.
        returns = ReturnSpec(modes=ReturnMode.POST_CYPHER) if return_rows else None
        ctx: Dict[str, Any] = {"db_path": self._db_path, "label": label}
        if pk_fields:
            ctx["pk_fields"] = pk_fields
        spec = OperationSpec(
            kind=OpKind.CREATE_NODE,
            label=label,
            props={"rows": rows},
            returns=returns,
            context=ctx,
        )
        tk = self._handler.submit(spec)
        try:
            res = tk.result(None)
        except Exception as e:
            raise RuntimeError(f"Execute failed: {e}") from e
        if return_rows and isinstance(res, dict):
            cr = res.get("cypher_results")
            if isinstance(cr, list) and cr and isinstance(cr[0], list):
                return cr[0]
            return []
        return None

    def update_node(self, *, label: str, pk: Dict[str, Any], props: Dict[str, Any]) -> None:
        """Update a node by label and primary key fields with provided properties via ATP."""
        if not isinstance(pk, dict) or not pk:
            raise ValueError("pk must be a non-empty dict of primary key fields")
        if not isinstance(props, dict) or not props:
            return  # nothing to update
        # Delegate update to ATP with structured operation
        spec = OperationSpec(
            kind=OpKind.UPDATE_NODE,
            label=label,
            pk=pk,
            props=props,
            returns=None,
            context={"db_path": self._db_path, "label": label},
        )
        tk = self._handler.submit(spec)
        try:
            _ = tk.result(None)
        except Exception as e:
            raise RuntimeError(f"Execute failed: {e}") from e

    def delete_node(self, *, label: str, pk: Dict[str, Any]) -> None:
        """Delete a node by label and primary key fields via ATP."""
        if not isinstance(pk, dict) or not pk:
            raise ValueError("pk must be a non-empty dict of primary key fields")
        # Delegate delete to ATP with structured operation
        spec = OperationSpec(
            kind=OpKind.DELETE_NODE,
            label=label,
            pk=pk,
            props={},
            returns=None,
            context={"db_path": self._db_path, "label": label},
        )
        tk = self._handler.submit(spec)
        try:
            _ = tk.result(None)
        except Exception as e:
            raise RuntimeError(f"Execute failed: {e}") from e

    def submit_many_specs(self, specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Submit many pre-built OperationSpec dicts through ATP and return their results in order.

        The dicts must follow OperationSpec schema (see ATP docs). This method blocks until all
        tickets complete, raising RuntimeError on the first failure observed.
        """
        if not isinstance(specs, list) or not all(isinstance(s, dict) for s in specs):
            raise ValueError("specs must be a list of dicts following OperationSpec schema")
        try:
            tickets = self._handler.submit_many_dict(specs)
        except Exception as e:
            raise RuntimeError(f"Submit-many failed: {e}") from e
        results: List[Dict[str, Any]] = []
        for t in tickets:
            try:
                res = t.result(None)
            except Exception as e:
                raise RuntimeError(f"Execute failed: {e}") from e
            results.append(res)
        return results

    def create_edges(
        self,
        *,
        rel_name: str,
        src_label: str,
        dst_label: str,
        rows: List[Dict[str, Any]],
        src_pk_field: str,
        dst_pk_field: str,
    ) -> Optional[List[Dict[str, Any]]]:
        # Bulk relationship create using typed rows with from_pk/to_pk + props
        returns = None
        if any("__atp_row_idx" in r for r in rows):
            # Request rows only when client provided per-row index for mapping
            returns = ReturnSpec(modes=ReturnMode.POST_CYPHER)
        # Build context; omit fixed labels/pk fields when wildcard '*' provided to enable dynamic grouping in Rust
        ctx: Dict[str, Any] = {"db_path": self._db_path, "rel_type": rel_name}
        if src_label and src_label != "*":
            ctx["src_label"] = src_label
        if dst_label and dst_label != "*":
            ctx["dst_label"] = dst_label
        if src_pk_field and src_pk_field != "*":
            ctx["src_pk_field"] = src_pk_field
        if dst_pk_field and dst_pk_field != "*":
            ctx["dst_pk_field"] = dst_pk_field

        spec = OperationSpec(
            kind=OpKind.CREATE_EDGE,
            rel_type=rel_name,
            src=(src_label, {}),
            dst=(dst_label, {}),
            props={"rows": rows},
            returns=returns,
            context=ctx,
        )
        tk = self._handler.submit(spec)
        try:
            res = tk.result(None)
        except Exception as e:
            raise RuntimeError(f"Execute failed: {e}") from e
        if returns is not None and isinstance(res, dict):
            cr = res.get("cypher_results")
            if isinstance(cr, list) and cr and isinstance(cr[0], list):
                return cr[0]
            return []
        return None

    def flush(self, timeout: Optional[float] = None) -> None:
        self._handler.flush(timeout)

    def release(self, timeout: Optional[float] = None) -> None:
        if self._ephemeral_handler is not None:
            try:
                self._ephemeral_handler.flush(timeout)
            finally:
                self._ephemeral_handler.shutdown(timeout)
            self._ephemeral_handler = None
            return
        _release_handler(self._db_path, timeout)

    def shutdown(self, timeout: Optional[float] = None) -> None:
        # Backward-compat alias
        self.release(timeout)

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import threading

from atp_pipeline import (
    ATPHandler,
    DatabaseType,
    OperationSpec,
    ReturnSpec,
    OpKind,
    ReturnMode,
)


_HANDLER_LOCK = threading.RLock()
_HANDLERS: Dict[str, Tuple[ATPHandler, int]] = {}


def _acquire_handler(db_path: str) -> ATPHandler:
    key = str(Path(db_path).resolve())
    with _HANDLER_LOCK:
        if key in _HANDLERS:
            handler, refc = _HANDLERS[key]
            _HANDLERS[key] = (handler, refc + 1)
            return handler
        # Create new handler and store with refcount 1
        cfg: Dict[str, Any] = {"db_path": key}
        handler: ATPHandler = ATPHandler(DatabaseType.KUZU, cfg)
        # Force capability negotiation early for explicit failure
        _ = handler.get_capability_report()
        _HANDLERS[key] = (handler, 1)
        return handler


def _release_handler(db_path: str, timeout: Optional[float]) -> None:
    key = str(Path(db_path).resolve())
    with _HANDLER_LOCK:
        entry = _HANDLERS.get(key)
        if not entry:
            return
        handler, refc = entry
        if refc > 1:
            _HANDLERS[key] = (handler, refc - 1)
            return
        # Last reference: flush and shutdown, then remove
        try:
            handler.flush(timeout)
        finally:
            handler.shutdown(timeout)
        _HANDLERS.pop(key, None)


class ATPIntegration:
    """Integration with ATP pipeline for Kuzu database operations."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(Path(db_path).resolve())
        # Acquire or create a shared handler per db_path
        self._handler: ATPHandler = _acquire_handler(self._db_path)

    def run_cypher(self, query: str, params: Optional[Dict[str, Any]], expect_rows: bool) -> List[Dict[str, Any]]:
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
        _release_handler(self._db_path, timeout)

    def shutdown(self, timeout: Optional[float] = None) -> None:
        # Backward-compat alias
        self.release(timeout)

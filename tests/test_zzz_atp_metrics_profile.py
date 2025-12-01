from __future__ import annotations

from pathlib import Path
import os

from atp_pipeline import ATPHandler, DatabaseType, OperationSpec, ReturnSpec, OpKind, ReturnMode
import atp_pipeline._atp_pipeline as _ext


def test_dump_atp_metrics_snapshot() -> None:
    """Ensure Rust observability metrics are exposed and persist them for offline analysis.

    This test exercises the metrics_snapshot API at the end of the full test run and writes the
    Prometheus text exposition to a stable file in the repository root. The file is intended for
    profiling and should be inspected after running the full test suite.
    """
    handler = ATPHandler(DatabaseType.KUZU, {"db_path": ":memory:"})

    # Optionally enable CPU profiling if compiled; for flame-based profiling
    # this is a no-op that simply logs intent while ATP_PROFILE=1 controls
    # whether env-driven dumps are wired up.
    handler.start_profiling(99)

    # Execute a small but real ATP operation to ensure that profiling spans
    # are recorded on the executor thread. This uses a trivial RUN_CYPHER
    # operation against the in-memory Kuzu database.
    spec = OperationSpec(
        kind=OpKind.RUN_CYPHER,
        cypher=["RETURN 1"],
        props={},
        returns=ReturnSpec(modes=ReturnMode.POST_CYPHER),
    )
    tk = handler.submit(spec)
    _ = tk.result(None)

    snapshot = handler.metrics_snapshot()

    root = Path(__file__).resolve().parents[1]
    out_path = root / "atp_metrics_snapshot.prom"
    out_path.write_text(snapshot, encoding="utf-8")

    # Basic invariant: API is callable and returns a string payload.
    assert isinstance(snapshot, str)

    # Cleanly shut down the handler so that executor threads drain and commit
    # their flame spans before we trigger any dumps.
    handler.shutdown(None)

    # Optionally dump profiler outputs if supported by the build
    handler.dump_profiling_flamegraph(str(root / "atp_flame.svg"))
    handler.dump_profiling_protobuf(str(root / "atp_profile.pb"))

    # When ATP_PROFILE=1 is set (e.g. during full pytest runs with profiling
    # enabled), also honour the env-controlled profiling outputs by invoking
    # the native dump_profiling_artifacts_env hook explicitly. This ensures
    # that Speedscope JSON and flamegraph artifacts are written to the paths
    # configured via ATP_PROFILE_SPEEDSCOPE and ATP_PROFILE_FLAMEGRAPH,
    # independent of Python's atexit behaviour.
    if os.environ.get("ATP_PROFILE") == "1":
        _ext.dump_profiling_artifacts_env()

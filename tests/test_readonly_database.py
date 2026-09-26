"""Verify physical database immutability through the ATP connection contract."""

from __future__ import annotations

import hashlib

import atp_pipeline
import pytest
from kuzualchemy import KuzuConnection


def test_readonly_database_preserves_bytes_and_denies_write(tmp_path, monkeypatch):
    for key, value in {"ATP_BUFFER_POOL_BYTES": str(64 * 2**20), "ATP_KUZU_MAX_DB_SIZE_BYTES": str(2 * 2**30),
                       "ATP_MAX_THREADS": "1", "ATP_READONLY_POOL_MAX_SIZE": "1", "ATP_WRITE_POOL_MAX_SIZE": "1"}.items():
        monkeypatch.setenv(key, value)
    path = tmp_path / "retained.kuzu"
    writer = KuzuConnection(path)
    try:
        writer.execute_write("CREATE NODE TABLE Item(id INT64, PRIMARY KEY(id))")
        writer.execute_write("CREATE (:Item {id: 41})")
    finally:
        writer.close()
        atp_pipeline.dispose_kuzu_database(str(path))
    before = hashlib.sha256(path.read_bytes()).digest()
    path.chmod(0o400)
    for _ in range(2):
        reader = KuzuConnection(path, read_only=True)
        try:
            assert reader.execute("MATCH (n:Item) RETURN n.id AS id LIMIT 1") == [{"id": 41}]
            with pytest.raises((ValueError, RuntimeError), match="read.only"):
                reader.execute_write("CREATE (:Item {id: 42})")
        finally:
            reader.close()
            atp_pipeline.dispose_kuzu_database(str(path))
        assert hashlib.sha256(path.read_bytes()).digest() == before


@pytest.mark.parametrize("value", [1, "true", None])
def test_readonly_mode_requires_boolean(tmp_path, value):
    with pytest.raises(TypeError, match="boolean"):
        KuzuConnection(tmp_path / "graph.kuzu", read_only=value)


def test_missing_native_readonly_capability_rejects_before_database_access(tmp_path, monkeypatch):
    from kuzualchemy import kuzu_connection

    handlers = []
    class Handler:
        def __init__(self, database, config):
            self.config, self.closed = config, False
            handlers.append(self)
        def get_capability_report(self):
            return {"capabilities": "EngineCapabilities(CYPHER | TRANSACTIONS)"}
        def shutdown(self):
            self.closed = True
    monkeypatch.setattr(kuzu_connection, "ATPHandler", Handler)
    path = tmp_path / "absent.kuzu"
    with pytest.raises(RuntimeError, match="lacks read-only"):
        KuzuConnection(path, read_only=True)
    assert handlers[0].config["read_only"] is True
    assert handlers[0].closed and not path.exists()

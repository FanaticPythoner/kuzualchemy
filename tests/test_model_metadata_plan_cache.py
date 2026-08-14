from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock, Thread
from typing import Any

from kuzualchemy import KuzuDataType, KuzuNodeBase, kuzu_field, kuzu_node
from kuzualchemy.kuzu_orm import (
    _apply_kuzu_field_directives,
    _kuzu_registry,
    clear_model_kuzu_metadata_cache,
    kuzu_field_edit,
)


def _metadata_plan_node(name: str) -> type[KuzuNodeBase]:
    @kuzu_node(name, abstract=True)
    class MetadataPlanNode(KuzuNodeBase):
        id: int | None = kuzu_field(
            kuzu_type=KuzuDataType.INT64,
            primary_key=True,
            auto_increment=True,
        )
        name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)

    return MetadataPlanNode


def test_model_metadata_plan_scans_fields_once_and_returns_copy_safe_collections(monkeypatch: Any) -> None:
    node_class = _metadata_plan_node("MetadataPlanCopySafeNode")
    clear_model_kuzu_metadata_cache()
    original_get_field_metadata = _kuzu_registry.get_field_metadata
    metadata_calls = 0

    def counted_get_field_metadata(field_info: Any) -> Any:
        nonlocal metadata_calls
        metadata_calls += 1
        return original_get_field_metadata(field_info)

    monkeypatch.setattr(_kuzu_registry, "get_field_metadata", counted_get_field_metadata)

    for node_id in range(100):
        node_class(id=node_id, name=str(node_id))

    primary_keys = node_class.get_primary_key_fields()
    primary_keys.append("mutated")
    auto_increment_fields = node_class.get_auto_increment_fields()
    auto_increment_fields.append("mutated")
    all_metadata = node_class.get_all_kuzu_metadata()
    all_metadata.clear()
    auto_increment_metadata = node_class.get_auto_increment_metadata()
    auto_increment_metadata.clear()

    assert metadata_calls == len(node_class.model_fields)
    assert node_class.get_primary_key_fields() == ["id"]
    assert node_class.get_auto_increment_fields() == ["id"]
    assert set(node_class.get_all_kuzu_metadata()) == {"id", "name"}
    assert set(node_class.get_auto_increment_metadata()) == {"id"}
    assert node_class.has_auto_increment_primary_key()


def test_model_metadata_plan_first_access_is_deterministic_under_parallelism(monkeypatch: Any) -> None:
    node_class = _metadata_plan_node("MetadataPlanParallelNode")
    clear_model_kuzu_metadata_cache()
    original_get_field_metadata = _kuzu_registry.get_field_metadata
    metadata_calls = 0
    counter_lock = Lock()
    worker_count = 32
    barrier = Barrier(worker_count)

    def counted_get_field_metadata(field_info: Any) -> Any:
        nonlocal metadata_calls
        with counter_lock:
            metadata_calls += 1
        return original_get_field_metadata(field_info)

    def construct_node(node_id: int) -> tuple[int | None, tuple[str, ...], tuple[str, ...]]:
        barrier.wait()
        node = node_class(id=node_id, name=str(node_id))
        return (
            node.id,
            tuple(node_class.get_primary_key_fields()),
            tuple(node_class.get_auto_increment_fields()),
        )

    monkeypatch.setattr(_kuzu_registry, "get_field_metadata", counted_get_field_metadata)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(construct_node, range(worker_count)))

    assert results == [(node_id, ("id",), ("id",)) for node_id in range(worker_count)]
    assert metadata_calls == len(node_class.model_fields)


def test_model_metadata_plan_invalidates_after_field_directive_mutation() -> None:
    class MetadataPlanMutableNode(KuzuNodeBase):
        id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)

    assert not MetadataPlanMutableNode.has_auto_increment_primary_key()

    _apply_kuzu_field_directives(
        MetadataPlanMutableNode,
        [kuzu_field_edit("id", auto_increment=True)],
    )

    assert MetadataPlanMutableNode.get_primary_key_fields() == ["id"]
    assert MetadataPlanMutableNode.get_auto_increment_fields() == ["id"]
    assert MetadataPlanMutableNode.has_auto_increment_primary_key()


def test_model_metadata_plan_invalidation_serializes_with_active_build(monkeypatch: Any) -> None:
    node_class = _metadata_plan_node("MetadataPlanInvalidationNode")
    clear_model_kuzu_metadata_cache()
    original_get_field_metadata = _kuzu_registry.get_field_metadata
    first_scan_started = Event()
    release_first_scan = Event()
    metadata_calls = 0
    counter_lock = Lock()

    def blocked_get_field_metadata(field_info: Any) -> Any:
        nonlocal metadata_calls
        with counter_lock:
            metadata_calls += 1
            current_call = metadata_calls
        if current_call == 1:
            first_scan_started.set()
            assert release_first_scan.wait(timeout=5.0)
        return original_get_field_metadata(field_info)

    monkeypatch.setattr(_kuzu_registry, "get_field_metadata", blocked_get_field_metadata)
    builder = Thread(target=node_class.get_primary_key_fields)
    builder.start()
    assert first_scan_started.wait(timeout=5.0)

    invalidation_complete = Event()

    def invalidate() -> None:
        clear_model_kuzu_metadata_cache()
        invalidation_complete.set()

    invalidator = Thread(target=invalidate)
    invalidator.start()
    assert not invalidation_complete.wait(timeout=0.05)
    release_first_scan.set()
    builder.join(timeout=5.0)
    invalidator.join(timeout=5.0)

    assert not builder.is_alive()
    assert not invalidator.is_alive()
    assert invalidation_complete.is_set()
    first_build_calls = metadata_calls
    assert node_class.get_primary_key_fields() == ["id"]
    assert metadata_calls == first_build_calls + len(node_class.model_fields)

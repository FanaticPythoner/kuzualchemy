from __future__ import annotations

import uuid
from typing import ClassVar

import pytest
from atp_pipeline import DbBulkAction
from pydantic import BaseModel, model_validator

from kuzualchemy.kuzu_relationship_read import construct_model_from_db_payload
from kuzualchemy.kuzu_session import KuzuSession
from kuzualchemy.kuzu_session_rows import (
    _primary_key_fields,
    _relationship_pair_metadata,
    clear_session_row_metadata_caches,
    model_field_specs,
    node_merge_policies,
    normalize_model_row,
)


class _Metadata:
    def __init__(
        self,
        *,
        kuzu_type: str,
        primary_key: bool = False,
        not_null: bool = False,
        auto_increment: bool = False,
        atp_merge_policy: str | None = None,
    ) -> None:
        self.kuzu_type = kuzu_type
        self.primary_key = primary_key
        self.not_null = not_null
        self.auto_increment = auto_increment
        self.atp_merge_policy = atp_merge_policy


class _CachedNode:
    __kuzu_node_name__ = "CachedNode"
    metadata_calls = 0
    primary_key_calls = 0

    def __init__(self, id: int) -> None:
        self.id = id

    @classmethod
    def get_all_kuzu_metadata(cls) -> dict[str, _Metadata]:
        cls.metadata_calls += 1
        return {
            "id": _Metadata(kuzu_type="INT64", primary_key=True, not_null=True),
            "external_id": _Metadata(kuzu_type="UUID", atp_merge_policy="keep_existing"),
        }

    @classmethod
    def get_primary_key_fields(cls) -> list[str]:
        cls.primary_key_calls += 1
        return ["id"]


class _CachedTargetNode(_CachedNode):
    __kuzu_node_name__ = "CachedTargetNode"
    metadata_calls = 0
    primary_key_calls = 0


class _RelationshipPair:
    from_node = _CachedNode
    to_node = _CachedTargetNode

    def get_from_name(self) -> str:
        return "CachedNode"

    def get_to_name(self) -> str:
        return "CachedTargetNode"


class _CachedRelationship:
    __kuzu_rel_name__ = "CachedRelationship"
    __kuzu_relationship_pairs__ = [_RelationshipPair()]

    def __init__(self, from_node: _CachedNode, to_node: _CachedTargetNode, weight: int) -> None:
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight

    @classmethod
    def get_all_kuzu_metadata(cls) -> dict[str, _Metadata]:
        return {
            "weight": _Metadata(kuzu_type="INT64"),
        }


class _ConstructedModel(BaseModel):
    id: int
    name: str
    validator_calls: ClassVar[int] = 0

    @model_validator(mode="before")
    @classmethod
    def _count_validator(cls, values: object) -> object:
        cls.validator_calls += 1
        return values


class _CaptureConnection:
    db_path = ":memory:"

    def __init__(self) -> None:
        self.node_batches: list[tuple[object, ...]] = []
        self.relationship_batches: list[tuple[object, ...]] = []

    def bulk_write_nodes_and_relationships_many(
        self,
        node_batches: list[tuple[object, ...]],
        relationship_batches: list[tuple[object, ...]],
    ) -> None:
        self.node_batches.extend(node_batches)
        self.relationship_batches.extend(relationship_batches)


def _reset_counters() -> None:
    clear_session_row_metadata_caches()
    _CachedNode.metadata_calls = 0
    _CachedNode.primary_key_calls = 0
    _CachedTargetNode.metadata_calls = 0
    _CachedTargetNode.primary_key_calls = 0


def test_model_field_metadata_cache_is_class_scoped_and_copy_safe() -> None:
    _reset_counters()
    nil_uuid = uuid.UUID(int=0)

    for row_id in range(8):
        row = normalize_model_row(_CachedNode, {"id": row_id, "external_id": nil_uuid})
        assert row == {"id": row_id, "external_id": None}

    specs = model_field_specs(_CachedNode)
    specs[0]["field"] = "mutated"

    assert model_field_specs(_CachedNode)[0]["field"] == "id"
    assert _CachedNode.metadata_calls == 1


def test_primary_key_and_merge_policy_caches_return_mutable_copies() -> None:
    _reset_counters()

    fields = _primary_key_fields(_CachedNode)
    fields.append("mutated")
    policies = node_merge_policies(_CachedNode)
    policies["external_id"] = "mutated"

    assert _primary_key_fields(_CachedNode) == ["id"]
    assert node_merge_policies(_CachedNode) == {"external_id": "keep_existing"}
    assert _CachedNode.primary_key_calls == 1
    assert _CachedNode.metadata_calls == 1


def test_relationship_pair_metadata_reuses_endpoint_primary_key_metadata() -> None:
    _reset_counters()

    assert _relationship_pair_metadata(_CachedRelationship) == (
        ("CachedNode", "CachedTargetNode", "id", "id"),
    )
    assert _relationship_pair_metadata(_CachedRelationship) == (
        ("CachedNode", "CachedTargetNode", "id", "id"),
    )
    assert _CachedNode.primary_key_calls == 1
    assert _CachedTargetNode.primary_key_calls == 1


def test_session_identity_tracking_flag_disables_identity_map() -> None:
    _reset_counters()
    session = KuzuSession(connection=object(), identity_tracking=False)
    node = _CachedNode(1)

    session._remember(node)
    session.expire(node)

    assert session._identity_map == {}
    assert session._identity_keys_by_object_id == {}


def test_bulk_immediate_skips_identity_map_by_default() -> None:
    conn = _CaptureConnection()
    session = KuzuSession(connection=conn)
    node = _CachedNode(1)

    session.bulk_insert_immediate([node])

    assert session._identity_map == {}
    assert session._identity_keys_by_object_id == {}
    assert conn.node_batches


def test_unit_of_work_add_tracks_identity_by_default() -> None:
    session = KuzuSession(connection=_CaptureConnection(), expire_on_commit=False)
    node = _CachedNode(1)

    session.add(node)

    identity_key = session._identity_key(node)
    assert session._identity_map[identity_key] is node


def test_bulk_immediate_can_track_identity_when_requested() -> None:
    session = KuzuSession(connection=_CaptureConnection())
    node = _CachedNode(1)

    session.bulk_insert_immediate([node], track_identity=True)

    identity_key = session._identity_key(node)
    assert session._identity_map[identity_key] is node


def test_threaded_row_partition_preserves_order_and_identity(monkeypatch) -> None:
    _reset_counters()
    conn = _CaptureConnection()
    session = KuzuSession(connection=conn, bulk_batch_size=2)
    monkeypatch.setattr("kuzualchemy.kuzu_session.os.process_cpu_count", lambda: 2)
    target = _CachedTargetNode(100)
    nodes = [_CachedNode(row_id) for row_id in range(5)]
    relationships = [_CachedRelationship(nodes[row_id], target, row_id) for row_id in range(3)]

    assert session._row_partition_worker_count(len(nodes) + len(relationships)) == 2

    session._write_instance_batch(
        DbBulkAction.CREATE,
        [*nodes, *relationships],
        track_identity=True,
    )

    assert len(conn.node_batches) == 1
    assert [row["id"] for row in conn.node_batches[0][2]] == [0, 1, 2, 3, 4]
    assert len(conn.relationship_batches) == 1
    assert [row["weight"] for row in conn.relationship_batches[0][4]] == [0, 1, 2]
    assert [row["from_pk"] for row in conn.relationship_batches[0][4]] == [0, 1, 2]
    assert [session._identity_map[session._identity_key(node)] for node in nodes] == nodes


def test_row_partition_workers_are_limited_by_batch_size(monkeypatch) -> None:
    session = KuzuSession(connection=_CaptureConnection(), bulk_batch_size=4)
    monkeypatch.setattr("kuzualchemy.kuzu_session.os.process_cpu_count", lambda: 32)

    assert session._row_partition_worker_count(3) == 1
    assert session._row_partition_worker_count(10) == 3
    assert session._row_partition_worker_count(200) == 32


def test_graph_partition_rejects_misrouted_model_lists() -> None:
    conn = _CaptureConnection()
    session = KuzuSession(connection=conn)
    source = _CachedNode(1)
    target = _CachedTargetNode(2)
    relationship = _CachedRelationship(source, target, 3)

    with pytest.raises(TypeError, match="is not a registered Kuzu node"):
        session._write_instance_groups(
            DbBulkAction.CREATE,
            [relationship],
            [],
            track_identity=False,
        )
    with pytest.raises(TypeError, match="is not a registered Kuzu relationship"):
        session._write_instance_groups(
            DbBulkAction.CREATE,
            [],
            [source],
            track_identity=False,
        )


def test_db_payload_construction_bypasses_validation_and_filters_unknown_fields() -> None:
    _ConstructedModel.validator_calls = 0

    model = construct_model_from_db_payload(
        _ConstructedModel,
        {"id": 1, "name": "x", "_label": "ConstructedModel"},
    )

    assert model.id == 1
    assert model.name == "x"
    assert not hasattr(model, "_label")
    assert _ConstructedModel.validator_calls == 0

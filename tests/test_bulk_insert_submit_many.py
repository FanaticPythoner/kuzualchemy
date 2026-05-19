# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from kuzualchemy import (
    KuzuBaseModel,
    KuzuDataType,
    KuzuRelationshipBase,
    KuzuSession,
    kuzu_field,
    kuzu_node,
    kuzu_relationship,
)


@kuzu_node("BulkSubmitManyA")
class BulkSubmitManyA(KuzuBaseModel):
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)


@kuzu_node("BulkSubmitManyB")
class BulkSubmitManyB(KuzuBaseModel):
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)


@kuzu_relationship("BulkSubmitManyLinked", pairs=[(BulkSubmitManyA, BulkSubmitManyB)])
class BulkSubmitManyLinked(KuzuRelationshipBase):
    rank: int = kuzu_field(kuzu_type=KuzuDataType.INT64, default=0)


class _CaptureConnection:
    def __init__(self) -> None:
        self.node_writes: list[tuple[str, list[dict[str, Any]], list[str]]] = []
        self.relationship_writes: list[tuple[str, str, str, list[dict[str, Any]], list[str], list[str]]] = []
        self.relationship_reads: list[tuple[str, str, str, list[dict[str, str]], list[int]]] = []

    def bulk_write_nodes(
        self,
        _action: object,
        label: str,
        rows: list[dict[str, Any]],
        key_fields: list[str],
    ) -> None:
        self.node_writes.append((label, rows, key_fields))

    def bulk_write_relationships(
        self,
        _action: object,
        rel_type: str,
        from_label: str,
        to_label: str,
        rows: list[dict[str, Any]],
        from_key_fields: list[str],
        to_key_fields: list[str],
    ) -> None:
        self.relationship_writes.append(
            (rel_type, from_label, to_label, rows, from_key_fields, to_key_fields)
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
        self.relationship_reads.append((relationship, alias, direction, pairs, pairs_subset))
        return [
            {
                alias: {"rank": 7},
                "from_node": {"_label": "BulkSubmitManyA", "id": 1, "name": "a1"},
                "to_node": {"_label": "BulkSubmitManyB", "id": 2, "name": "b2"},
            }
        ]


def _session_for_capture(conn: _CaptureConnection) -> KuzuSession:
    session = KuzuSession.__new__(KuzuSession)
    session._conn = conn
    session._identity_map = {}
    session.autoflush = False
    session._new = []
    session._dirty = []
    session._deleted = []
    return session


def test_bulk_insert_submits_label_batches_by_model() -> None:
    conn = _CaptureConnection()
    session = _session_for_capture(conn)

    session.bulk_insert_immediate(
        [
            BulkSubmitManyA(id=3, name="a3"),
            BulkSubmitManyB(id=2, name="b2"),
            BulkSubmitManyA(id=1, name="a1"),
            BulkSubmitManyB(id=4, name="b4"),
        ],
        batch_size=1000,
    )

    labels = [entry[0] for entry in conn.node_writes]
    assert labels == ["BulkSubmitManyA", "BulkSubmitManyB"]
    a_rows = conn.node_writes[0][1]
    b_rows = conn.node_writes[1][1]
    assert [row["id"] for row in a_rows] == [3, 1]
    assert [row["id"] for row in b_rows] == [2, 4]


def test_bulk_insert_submits_concrete_relationship_routes() -> None:
    conn = _CaptureConnection()
    session = _session_for_capture(conn)

    session.bulk_insert_immediate([BulkSubmitManyLinked(from_node=1, to_node=2, rank=7)])

    assert conn.relationship_writes == [
        (
            "BulkSubmitManyLinked",
            "BulkSubmitManyA",
            "BulkSubmitManyB",
            [
                {
                    "rank": 7,
                    "from_label": "BulkSubmitManyA",
                    "to_label": "BulkSubmitManyB",
                    "from_pk_field": "id",
                    "to_pk_field": "id",
                    "from_pk": 1,
                    "to_pk": 2,
                }
            ],
            ["id"],
            ["id"],
        )
    ]


def test_relationship_query_submits_metadata_to_atp_relationship_read() -> None:
    conn = _CaptureConnection()
    session = _session_for_capture(conn)

    rows = session.query(BulkSubmitManyLinked).all()

    assert [row.rank for row in rows] == [7]
    assert [(row.from_node, row.to_node) for row in rows] == [(1, 2)]
    assert conn.relationship_reads == [
        (
            "BulkSubmitManyLinked",
            "n",
            "forward",
            [
                {
                    "from_label": "BulkSubmitManyA",
                    "to_label": "BulkSubmitManyB",
                    "from_key_field": "id",
                    "to_key_field": "id",
                }
            ],
            [],
        )
    ]


def test_bulk_update_relationships_submits_concrete_routes() -> None:
    conn = _CaptureConnection()
    session = _session_for_capture(conn)

    session.bulk_update_relationships(
        [BulkSubmitManyLinked(from_node=1, to_node=2, rank=11)],
        ["rank"],
    )

    assert conn.relationship_writes == [
        (
            "BulkSubmitManyLinked",
            "BulkSubmitManyA",
            "BulkSubmitManyB",
            [
                {
                    "from_label": "BulkSubmitManyA",
                    "to_label": "BulkSubmitManyB",
                    "from_pk_field": "id",
                    "to_pk_field": "id",
                    "from_pk": 1,
                    "to_pk": 2,
                    "rank": 11,
                }
            ],
            ["id"],
            ["id"],
        )
    ]

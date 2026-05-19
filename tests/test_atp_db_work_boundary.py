# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from kuzualchemy import (
    KuzuBaseModel,
    KuzuDataType,
    KuzuRelationshipBase,
    KuzuSession,
    get_ddl_for_node,
    get_ddl_for_relationship,
    kuzu_field,
    kuzu_node,
    kuzu_relationship,
)
from kuzualchemy.test_utilities import initialize_schema


@kuzu_node("BoundaryAuthor")
class BoundaryAuthor(KuzuBaseModel):
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
    score: int = kuzu_field(kuzu_type=KuzuDataType.INT64, default=0)


@kuzu_node("BoundaryPost")
class BoundaryPost(KuzuBaseModel):
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    title: str = kuzu_field(kuzu_type=KuzuDataType.STRING)


@kuzu_relationship("BoundaryAuthored", pairs=[(BoundaryAuthor, BoundaryPost)])
class BoundaryAuthored(KuzuRelationshipBase):
    rank: int = kuzu_field(kuzu_type=KuzuDataType.INT64, default=0)
    marker: str | None = kuzu_field(kuzu_type=KuzuDataType.STRING, default=None, not_null=False)


def _session(tmp_path: Path) -> KuzuSession:
    session = KuzuSession(db_path=tmp_path / "db", bulk_insert_threshold=1)
    initialize_schema(
        session,
        ddl="\n".join(
            [
                get_ddl_for_node(BoundaryAuthor),
                get_ddl_for_node(BoundaryPost),
                get_ddl_for_relationship(BoundaryAuthored),
            ]
        ),
    )
    return session


def test_atp_db_work_updates_and_deletes_nodes(tmp_path: Path) -> None:
    session = _session(tmp_path)
    try:
        session.bulk_insert_immediate(
            [
                BoundaryAuthor(id=1, name="alpha", score=1),
                BoundaryAuthor(id=2, name="beta", score=2),
            ]
        )

        session.bulk_update_nodes(
            BoundaryAuthor,
            [
                {"id": 1, "name": "alpha-new", "score": 11},
                {"id": 2, "name": "beta-new", "score": 22},
            ],
        )
        session.bulk_delete_nodes(BoundaryAuthor, [2])

        rows = session.execute(
            "MATCH (a:BoundaryAuthor) RETURN a.id AS id, a.name AS name, a.score AS score ORDER BY id"
        )

        assert rows == [{"id": 1, "name": "alpha-new", "score": 11}]
    finally:
        session.close()


def test_atp_db_work_updates_relationships(tmp_path: Path) -> None:
    session = _session(tmp_path)
    try:
        author = BoundaryAuthor(id=1, name="alpha", score=1)
        post = BoundaryPost(id=10, title="first")
        session.bulk_insert_immediate([author, post])
        session.bulk_insert_immediate(
            [
                BoundaryAuthored.create_between(
                    author.id,
                    post.id,
                    rank=1,
                    marker="before",
                )
            ]
        )

        session.bulk_update_relationships(
            [
                BoundaryAuthored.create_between(
                    author.id,
                    post.id,
                    rank=9,
                    marker=None,
                )
            ],
            ["rank", "marker"],
        )

        rows = session.execute(
            "MATCH (:BoundaryAuthor)-[r:BoundaryAuthored]->(:BoundaryPost) "
            "RETURN r.rank AS rank, r.marker AS marker"
        )

        assert rows == [{"rank": 9, "marker": None}]

        session.delete(
            BoundaryAuthored.create_between(
                author.id,
                post.id,
                rank=9,
                marker=None,
            )
        )
        session.flush()

        rows_after_delete = session.execute(
            "MATCH (:BoundaryAuthor)-[r:BoundaryAuthored]->(:BoundaryPost) RETURN count(r) AS count"
        )

        assert rows_after_delete == [{"count": 0}]
    finally:
        session.close()


def test_atp_relationship_read_uses_native_route_metadata(tmp_path: Path) -> None:
    session = _session(tmp_path)
    try:
        author = BoundaryAuthor(id=1, name="alpha", score=1)
        post = BoundaryPost(id=10, title="first")
        session.bulk_insert_immediate([author, post])
        session.bulk_insert_immediate(
            [
                BoundaryAuthored.create_between(
                    author.id,
                    post.id,
                    rank=3,
                    marker="route",
                )
            ]
        )

        rels = session.query(BoundaryAuthored).all()

        assert [(rel.rank, rel.marker, rel.from_node, rel.to_node) for rel in rels] == [
            (3, "route", 1, 10)
        ]
    finally:
        session.close()

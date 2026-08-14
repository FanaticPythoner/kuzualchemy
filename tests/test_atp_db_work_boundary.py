# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from kuzualchemy import (
    KuzuBaseModel,
    KuzuDataType,
    KuzuNodeBase,
    KuzuRelationshipBase,
    KuzuSession,
    get_ddl_for_node,
    get_ddl_for_relationship,
    kuzu_field,
    kuzu_node,
    kuzu_relationship,
)
from kuzualchemy.kuzu_session_rows import _node_row, _relationship_row, node_merge_policies
from kuzualchemy.test_utilities import initialize_schema


@kuzu_node("BoundaryAuthor")
class BoundaryAuthor(KuzuBaseModel):
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
    score: int = kuzu_field(
        kuzu_type=KuzuDataType.INT64,
        default=0,
        atp_merge_policy="KEEP_MAX_NUMERIC",
    )
    nickname: str | None = kuzu_field(kuzu_type=KuzuDataType.STRING, default=None, not_null=False)


@kuzu_node("BoundaryPost")
class BoundaryPost(KuzuBaseModel):
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    title: str = kuzu_field(kuzu_type=KuzuDataType.STRING)


@kuzu_relationship("BoundaryAuthored", pairs=[(BoundaryAuthor, BoundaryPost)])
class BoundaryAuthored(KuzuRelationshipBase):
    rank: int = kuzu_field(kuzu_type=KuzuDataType.INT64, default=0)
    marker: str | None = kuzu_field(kuzu_type=KuzuDataType.STRING, default=None, not_null=False)


@kuzu_relationship(
    "BoundaryLinked",
    pairs=[(BoundaryAuthor, BoundaryPost), (BoundaryPost, BoundaryAuthor)],
)
class BoundaryLinked(KuzuRelationshipBase):
    weight: int = kuzu_field(kuzu_type=KuzuDataType.INT64, default=0)


@kuzu_node("BoundaryEndpointAuthor")
class BoundaryEndpointAuthor(KuzuNodeBase):
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)


@kuzu_node("BoundaryEndpointPost")
class BoundaryEndpointPost(KuzuNodeBase):
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    title: str = kuzu_field(kuzu_type=KuzuDataType.STRING)


@kuzu_relationship(
    "BoundaryEndpointLinked",
    pairs=[
        (BoundaryEndpointAuthor, BoundaryEndpointPost),
        (BoundaryEndpointPost, BoundaryEndpointAuthor),
    ],
)
class BoundaryEndpointLinked(KuzuRelationshipBase):
    weight: int = kuzu_field(kuzu_type=KuzuDataType.INT64, default=0)


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


def test_atp_db_work_node_create_is_idempotent_without_null_overwrite(tmp_path: Path) -> None:
    session = _session(tmp_path)
    try:
        session.add_all(
            [
                BoundaryAuthor(id=1, name="alpha", score=1, nickname="live"),
                BoundaryAuthor(id=1, name="alpha", score=1, nickname=None),
            ]
        )
        session.flush()
        session.add(BoundaryAuthor(id=1, name="alpha", score=1, nickname=None))
        session.flush()

        rows = session.execute(
            "MATCH (a:BoundaryAuthor) RETURN a.id AS id, a.name AS name, "
            "a.score AS score, a.nickname AS nickname ORDER BY id"
        )

        assert rows == [{"id": 1, "name": "alpha", "score": 1, "nickname": "live"}]
    finally:
        session.close()


def test_bulk_row_serialization_matches_pydantic_contract() -> None:
    author = BoundaryAuthor(id=1, name="alpha", score=1)
    post = BoundaryPost(id=10, title="first")
    rel = BoundaryAuthored.create_between(
        author.id,
        post.id,
        rank=3,
        marker=None,
    )
    expected_relationship = rel.model_dump(
        mode="python",
        exclude={"from_node", "to_node"},
    )
    expected_relationship.update(
        {
            "from_label": "BoundaryAuthor",
            "to_label": "BoundaryPost",
            "from_pk_field": "id",
            "to_pk_field": "id",
            "from_pk": author.id,
            "to_pk": post.id,
        }
    )

    assert _node_row(author) == author.model_dump(mode="python")
    assert node_merge_policies(BoundaryAuthor) == {"score": "KEEP_MAX_NUMERIC"}
    assert _relationship_row(rel) == expected_relationship


def test_atp_merge_policy_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="atp_merge_policy must be one of"):
        kuzu_field(kuzu_type=KuzuDataType.INT64, atp_merge_policy="KEEP_LAST")

    with pytest.raises(TypeError, match="atp_merge_policy must be a string"):
        kuzu_field(kuzu_type=KuzuDataType.INT64, atp_merge_policy=object())  # type: ignore[arg-type]


def test_bulk_row_serialization_resolves_multi_pair_object_endpoints() -> None:
    author = BoundaryEndpointAuthor(id=1, name="alpha")
    post = BoundaryEndpointPost(id=10, title="first")
    rel = BoundaryEndpointLinked.create_between(author, post, weight=7)

    assert _relationship_row(rel) == {
        "weight": 7,
        "from_label": "BoundaryEndpointAuthor",
        "to_label": "BoundaryEndpointPost",
        "from_pk_field": "id",
        "to_pk_field": "id",
        "from_pk": author.id,
        "to_pk": post.id,
    }


def test_bulk_row_serialization_rejects_ambiguous_multi_pair_raw_endpoints() -> None:
    rel = BoundaryLinked.create_between(1, 10, weight=7)

    with pytest.raises(ValueError, match="BoundaryLinked endpoint route is ambiguous"):
        _relationship_row(rel)


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


def test_atp_relationship_iterator_pages_duplicate_endpoints_exactly_once(
    tmp_path: Path,
) -> None:
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
                    marker="first",
                ),
                BoundaryAuthored.create_between(
                    author.id,
                    post.id,
                    rank=2,
                    marker="second",
                ),
            ]
        )

        relationships = list(
            session.query(BoundaryAuthored).iter(
                page_size=1,
                prefetch_pages=1,
            )
        )

        assert sorted(
            (relationship.rank, relationship.marker) for relationship in relationships
        ) == [(1, "first"), (2, "second")]
        assert all(
            (relationship.from_node, relationship.to_node) == (author.id, post.id)
            for relationship in relationships
        )
    finally:
        session.close()

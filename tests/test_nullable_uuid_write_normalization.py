# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

import pytest

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


@kuzu_node("NilUuidParent")
class NilUuidParent(KuzuBaseModel):
    id: uuid.UUID = kuzu_field(kuzu_type=KuzuDataType.UUID, primary_key=True)
    nullable_uuid: Optional[uuid.UUID] = kuzu_field(
        default=None,
        kuzu_type=KuzuDataType.UUID,
        not_null=False,
    )
    pep604_uuid: uuid.UUID | None = kuzu_field(
        default=None,
        kuzu_type=KuzuDataType.UUID,
        not_null=False,
    )


@kuzu_node("NilUuidChild")
class NilUuidChild(KuzuBaseModel):
    id: uuid.UUID = kuzu_field(kuzu_type=KuzuDataType.UUID, primary_key=True)


@kuzu_relationship("NilUuidReference", pairs=[(NilUuidParent, NilUuidChild)])
class NilUuidReference(KuzuRelationshipBase):
    nullable_uuid: Optional[uuid.UUID] = kuzu_field(
        default=None,
        kuzu_type=KuzuDataType.UUID,
        not_null=False,
    )
    strict_uuid: uuid.UUID = kuzu_field(kuzu_type=KuzuDataType.UUID, not_null=True)


def _session(tmp_path: Path) -> KuzuSession:
    session = KuzuSession(db_path=tmp_path / "db", bulk_insert_threshold=1)
    initialize_schema(
        session,
        ddl="\n".join(
            [
                get_ddl_for_node(NilUuidParent),
                get_ddl_for_node(NilUuidChild),
                get_ddl_for_relationship(NilUuidReference),
            ]
        ),
    )
    return session


def test_bulk_relationship_nullable_nil_uuid_persists_as_null(tmp_path: Path) -> None:
    session = _session(tmp_path)
    parent = NilUuidParent(id=uuid.uuid4())
    child = NilUuidChild(id=uuid.uuid4())
    strict_uuid = uuid.uuid4()

    session.bulk_insert_immediate([parent, child])
    session.bulk_insert_immediate(
        [
            NilUuidReference.create_between(
                parent.id,
                child.id,
                nullable_uuid=uuid.UUID(int=0),
                strict_uuid=strict_uuid,
            )
        ]
    )

    rows = session.execute(
        "MATCH (p:NilUuidParent)-[r:NilUuidReference]->(c:NilUuidChild) "
        "RETURN r.nullable_uuid AS nullable_uuid, r.strict_uuid AS strict_uuid"
    )

    assert rows == [{"nullable_uuid": None, "strict_uuid": strict_uuid}]


def test_bulk_node_insert_nullable_nil_uuid_persists_as_null(tmp_path: Path) -> None:
    session = _session(tmp_path)
    parent = NilUuidParent(
        id=uuid.uuid4(),
        nullable_uuid=uuid.UUID(int=0),
        pep604_uuid=uuid.UUID(int=0),
    )

    session.bulk_insert_immediate([parent])

    rows = session.execute(
        "MATCH (p:NilUuidParent) "
        "RETURN p.nullable_uuid AS nullable_uuid, p.pep604_uuid AS pep604_uuid"
    )

    assert rows == [{"nullable_uuid": None, "pep604_uuid": None}]


def test_bulk_node_update_nullable_nil_uuid_persists_as_null(tmp_path: Path) -> None:
    session = _session(tmp_path)
    parent = NilUuidParent(
        id=uuid.uuid4(),
        nullable_uuid=uuid.uuid4(),
        pep604_uuid=uuid.uuid4(),
    )
    session.bulk_insert_immediate([parent])

    session.bulk_update_nodes(
        NilUuidParent,
        [
            {
                "id": parent.id,
                "nullable_uuid": uuid.UUID(int=0),
                "pep604_uuid": uuid.UUID(int=0),
            }
        ],
    )

    rows = session.execute(
        "MATCH (p:NilUuidParent) "
        "RETURN p.nullable_uuid AS nullable_uuid, p.pep604_uuid AS pep604_uuid"
    )

    assert rows == [{"nullable_uuid": None, "pep604_uuid": None}]


def test_bulk_relationship_nonnull_nil_uuid_is_rejected(tmp_path: Path) -> None:
    session = _session(tmp_path)
    parent = NilUuidParent(id=uuid.uuid4())
    child = NilUuidChild(id=uuid.uuid4())
    relationship = NilUuidReference.create_between(
        parent.id,
        child.id,
        nullable_uuid=None,
        strict_uuid=uuid.UUID(int=0),
    )

    with pytest.raises(ValueError, match="strict_uuid"):
        session._build_rel_rows_fixed([relationship], has_auto_increment=False)

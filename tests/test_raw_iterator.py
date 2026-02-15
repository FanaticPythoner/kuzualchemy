# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tempfile
import shutil
import uuid
import re
from pathlib import Path

import pytest

from kuzualchemy import KuzuSession
from kuzualchemy.test_utilities import initialize_schema


def _new_isolated_session() -> tuple[KuzuSession, Path]:
    db_path = Path(tempfile.gettempdir()) / f"raw_iter_db_{uuid.uuid4().hex[:8]}"
    sess = KuzuSession(db_path=str(db_path))
    return sess, db_path


def _cleanup_session(sess: KuzuSession, db_path: Path) -> None:
    sess.close()
    shutil.rmtree(db_path, ignore_errors=True)


def _setup_basic_users(session, test_models, sample_users, sample_posts):
    # Re-register models cleared by autouse fixture
    from kuzualchemy.kuzu_orm import _kuzu_registry
    TestUser = test_models["TestUser"]
    TestPost = test_models["TestPost"]

    _kuzu_registry.register_node(TestUser.__kuzu_node_name__, TestUser)
    _kuzu_registry.register_node(TestPost.__kuzu_node_name__, TestPost)

    # Create schema
    initialize_schema(session)

    # Populate minimal data
    for u in sample_users:
        session.add(TestUser(**u))
    for p in sample_posts:
        session.add(TestPost(**p))
    session.commit()


def test_raw_iter_basic_ids(test_models, sample_users, sample_posts):
    sess, db_path = _new_isolated_session()
    try:
        _setup_basic_users(sess, test_models, sample_users, sample_posts)
        TestUser = test_models["TestUser"]
        label = TestUser.__kuzu_node_name__
        cypher = f"MATCH (n:{label}) RETURN n.id AS id ORDER BY id"

        # Eager baseline
        eager = sess.execute(cypher)
        ids_eager = [row["id"] for row in eager]

        # Iterator paging
        it = sess.iterate(cypher, page_size=2)
        ids_iter = [row["id"] for row in it]

        # Execute() with iterator
        ids_exec_iter = [row["id"] for row in sess.execute(cypher, as_iterator=True, page_size=2)]

        assert ids_iter == ids_eager == ids_exec_iter == [1, 2, 3]
    finally:
        _cleanup_session(sess, db_path)


def test_raw_iter_prefetch_equivalence(test_models, sample_users, sample_posts):
    sess, db_path = _new_isolated_session()
    try:
        _setup_basic_users(sess, test_models, sample_users, sample_posts)
        TestUser = test_models["TestUser"]
        label = TestUser.__kuzu_node_name__
        cypher = f"MATCH (n:{label}) RETURN n.id AS id ORDER BY id"

        a = [r["id"] for r in sess.iterate(cypher, page_size=2, prefetch_pages=0)]
        b = [r["id"] for r in sess.iterate(cypher, page_size=2, prefetch_pages=1)]
        c = [r["id"] for r in sess.execute(cypher, as_iterator=True, page_size=2, prefetch_pages=1)]
        assert a == b == c == [1, 2, 3]
    finally:
        _cleanup_session(sess, db_path)


def test_raw_iter_exact_multiple_page_size_avoids_terminal_out_of_range_query(
    test_models,
    sample_users,
    sample_posts,
    monkeypatch,
):
    sess, db_path = _new_isolated_session()
    try:
        _setup_basic_users(sess, test_models, sample_users, sample_posts)
        TestUser = test_models["TestUser"]
        label = TestUser.__kuzu_node_name__

        sess.add(TestUser(id=4, name="Dora", email="dora@example.com", age=22))
        sess.commit()

        executed_queries: list[str] = []
        original_execute = sess.execute

        def _spy_execute(query: str, parameters=None, **kwargs):
            executed_queries.append(query)
            return original_execute(query, parameters, **kwargs)

        monkeypatch.setattr(sess, "execute", _spy_execute)

        cypher = f"MATCH (n:{label}) RETURN n.id AS id ORDER BY id"
        ids_iter = [row["id"] for row in sess.iterate(cypher, page_size=2, prefetch_pages=1)]

        assert ids_iter == [1, 2, 3, 4]

        skip_values = [
            int(match.group(1))
            for q in executed_queries
            for match in [re.search(r"\bSKIP\s+(\d+)\b", q)]
            if match is not None
        ]
        assert skip_values == [0, 2]
    finally:
        _cleanup_session(sess, db_path)


def test_raw_iter_rejects_existing_skip_limit(test_models, sample_users, sample_posts):
    sess, db_path = _new_isolated_session()
    try:
        _setup_basic_users(sess, test_models, sample_users, sample_posts)
        TestUser = test_models["TestUser"]
        label = TestUser.__kuzu_node_name__
        cypher = f"MATCH (n:{label}) RETURN n.id AS id ORDER BY id LIMIT 10"
        with pytest.raises(ValueError):
            _ = list(sess.iterate(cypher, page_size=2))
        with pytest.raises(ValueError):
            _ = list(sess.execute(cypher, as_iterator=True, page_size=2))
    finally:
        _cleanup_session(sess, db_path)


def test_execute_as_iterator_default_pagesize_works(test_models, sample_users, sample_posts):
    sess, db_path = _new_isolated_session()
    try:
        _setup_basic_users(sess, test_models, sample_users, sample_posts)
        TestUser = test_models["TestUser"]
        label = TestUser.__kuzu_node_name__
        cypher = f"MATCH (n:{label}) RETURN n.id AS id ORDER BY id"
        # Should work without page_size (defaults to 10)
        ids = [row["id"] for row in sess.execute(cypher, as_iterator=True)]
        assert ids == [1, 2, 3]
        # Invalid page_size still rejected
        with pytest.raises(ValueError):
            _ = list(sess.execute(cypher, as_iterator=True, page_size=0))
    finally:
        _cleanup_session(sess, db_path)


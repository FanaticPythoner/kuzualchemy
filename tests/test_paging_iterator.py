# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re

import pytest
import tempfile
import shutil
import uuid
from pathlib import Path

from kuzualchemy import KuzuSession
from kuzualchemy.test_utilities import initialize_schema


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


def _new_isolated_session() -> tuple[KuzuSession, Path]:
    db_path = Path(tempfile.gettempdir()) / f"iter_db_{uuid.uuid4().hex[:8]}"
    sess = KuzuSession(db_path=str(db_path))
    return sess, db_path


def _cleanup_session(sess: KuzuSession, db_path: Path) -> None:
    sess.close()
    shutil.rmtree(db_path, ignore_errors=True)


def test_all_iterator_paging(test_models, sample_users, sample_posts):
    """Ensure .all(as_iterator=True, page_size=...) yields all items lazily across pages."""
    sess, db_path = _new_isolated_session()
    try:
        _setup_basic_users(sess, test_models, sample_users, sample_posts)
        TestUser = test_models["TestUser"]

        q = sess.query(TestUser).order_by("id")
        it = q.all(as_iterator=True, page_size=2)

        items = list(it)
        assert len(items) == 3
        assert all(isinstance(x, TestUser) for x in items)
    finally:
        _cleanup_session(sess, db_path)


def test_iter_convenience_method(test_models, sample_users, sample_posts):
    """Ensure Query.iter(page_size=...) behaves like the iterator form of .all()."""
    sess, db_path = _new_isolated_session()
    try:
        _setup_basic_users(sess, test_models, sample_users, sample_posts)
        TestUser = test_models["TestUser"]

        q = sess.query(TestUser).order_by("id")
        items = list(q.iter(page_size=2))

        assert len(items) == 3
        assert all(isinstance(x, TestUser) for x in items)
    finally:
        _cleanup_session(sess, db_path)


def test_all_default_behavior_unchanged(test_models, sample_users, sample_posts):
    """Calling .all() without flags should still return a list eagerly."""
    sess, db_path = _new_isolated_session()
    try:
        _setup_basic_users(sess, test_models, sample_users, sample_posts)
        TestUser = test_models["TestUser"]

        q = sess.query(TestUser)
        results = q.all()

        assert isinstance(results, list)
        assert all(isinstance(x, TestUser) for x in results)
    finally:
        _cleanup_session(sess, db_path)



def test_iter_prefetch_on_off_equivalence(test_models, sample_users, sample_posts):
    sess, db_path = _new_isolated_session()
    try:
        _setup_basic_users(sess, test_models, sample_users, sample_posts)
        TestUser = test_models["TestUser"]
        q = sess.query(TestUser).order_by("id")
        items_no_prefetch = list(q.iter(page_size=2, prefetch_pages=0))
        items_prefetch = list(q.iter(page_size=2, prefetch_pages=1))
        assert [u.id for u in items_no_prefetch] == [u.id for u in items_prefetch]
    finally:
        _cleanup_session(sess, db_path)


def test_iter_page_size_edge_cases(test_models, sample_users, sample_posts):
    sess, db_path = _new_isolated_session()
    try:
        _setup_basic_users(sess, test_models, sample_users, sample_posts)
        TestUser = test_models["TestUser"]

        # page_size = 1
        items_ps1 = list(sess.query(TestUser).order_by("id").iter(page_size=1))
        assert [u.id for u in items_ps1] == [1, 2, 3]

        # page_size larger than total
        items_large = list(sess.query(TestUser).order_by("id").iter(page_size=10))
        assert [u.id for u in items_large] == [1, 2, 3]
    finally:
        _cleanup_session(sess, db_path)


def test_iter_exact_multiple_page_size_avoids_terminal_out_of_range_query(
    test_models,
    sample_users,
    sample_posts,
    monkeypatch,
):
    sess, db_path = _new_isolated_session()
    try:
        _setup_basic_users(sess, test_models, sample_users, sample_posts)
        TestUser = test_models["TestUser"]
        # Expand to exact multiple of page_size=2 to exercise terminal-page behavior.
        sess.add(TestUser(id=4, name="Dora", email="dora@example.com", age=22))
        sess.commit()

        # Force sequential path for this regression check.
        monkeypatch.setenv("ATP_READONLY_POOL_MAX_SIZE", "1")

        executed_queries: list[str] = []
        original_execute = sess._execute_for_query_object

        def _spy_execute(query: str, parameters=None):
            executed_queries.append(query)
            return original_execute(query, parameters)

        monkeypatch.setattr(sess, "_execute_for_query_object", _spy_execute)

        items = list(sess.query(TestUser).order_by("id").iter(page_size=2, prefetch_pages=1))
        assert [u.id for u in items] == [1, 2, 3, 4]

        skip_values = [
            int(match.group(1))
            for q in executed_queries
            for match in [re.search(r"\bSKIP\s+(\d+)\b", q)]
            if match is not None
        ]
        assert skip_values == [0, 2]
    finally:
        _cleanup_session(sess, db_path)


def test_all_as_iterator_defaults_to_10_when_missing_page_size(test_models, sample_users, sample_posts):
    sess, db_path = _new_isolated_session()
    try:
        _setup_basic_users(sess, test_models, sample_users, sample_posts)
        TestUser = test_models["TestUser"]
        # Should work without page_size (defaults to 10)
        items = list(sess.query(TestUser).order_by("id").all(as_iterator=True))
        assert [u.id for u in items] == [1, 2, 3]
        # Invalid page_size still rejected
        with pytest.raises(ValueError):
            _ = sess.query(TestUser).all(as_iterator=True, page_size=0)
    finally:
        _cleanup_session(sess, db_path)


def test_iter_select_fields_returns_partial_models(test_models, sample_users, sample_posts):
    sess, db_path = _new_isolated_session()
    try:
        _setup_basic_users(sess, test_models, sample_users, sample_posts)
        TestUser = test_models["TestUser"]
        # Select only id and name; iterator yields partial model instances
        items = list(sess.query(TestUser).select("id", "name").order_by("id").iter(page_size=2))
        assert all(isinstance(row, TestUser) for row in items)
        assert [row.id for row in items] == [1, 2, 3]
        assert all(getattr(row, "name") is not None for row in items)
    finally:
        _cleanup_session(sess, db_path)


def test_iter_with_filter_and_order(test_models, sample_users):
    from kuzualchemy.kuzu_query_fields import QueryField
    sess, db_path = _new_isolated_session()
    try:
        _setup_basic_users(sess, test_models, sample_users, [])
        TestUser = test_models["TestUser"]
        age_field = QueryField("age", TestUser)
        items = list(
            sess.query(TestUser)
            .where(age_field >= 26)
            .order_by(age_field.desc())
            .iter(page_size=1)
        )
        ages = [u.age for u in items]
        assert ages == sorted(ages, reverse=True)
        assert all(a >= 26 for a in ages)
    finally:
        _cleanup_session(sess, db_path)


def test_iter_memory_bounded_for_large_dataset(test_models):
    import tracemalloc
    sess, db_path = _new_isolated_session()
    try:
        # Register models and create schema
        from kuzualchemy.kuzu_orm import _kuzu_registry
        TestUser = test_models["TestUser"]
        _kuzu_registry.register_node(TestUser.__kuzu_node_name__, TestUser)
        initialize_schema(sess)

        # Insert a few thousand users
        N = 2000
        for i in range(1, N + 1):
            sess.add(TestUser(id=i, name=f"User {i}", email=f"u{i}@x", age=i % 90))
        sess.commit()

        # Iterate with small page size and ensure memory doesn't balloon
        tracemalloc.start()
        _, start_peak = tracemalloc.get_traced_memory()
        count = 0
        for _ in sess.query(TestUser).order_by("id").iter(page_size=64, prefetch_pages=1):
            count += 1
        _, end_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert count == N
        # Peak additional memory during iteration should be under a conservative cap
        # Adjust if CI env differs significantly, but keep bounded
        added_peak = max(0, end_peak - start_peak)
        assert added_peak < 20 * 1024 * 1024  # < 20MB
    finally:
        _cleanup_session(sess, db_path)


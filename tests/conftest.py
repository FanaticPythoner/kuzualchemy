# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
Pytest configuration and shared fixtures for KuzuAlchemy tests.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Generator, Any, Dict, List
from unittest.mock import Mock, patch

import pytest

from . import _env  # noqa: F401  # pylint: disable=unused-import

from kuzualchemy import (
    kuzu_node,
    kuzu_relationship,
    KuzuBaseModel,
    kuzu_field,
    KuzuDataType,
    KuzuSession,
)
from kuzualchemy.constants import KuzuDefaultFunction
from kuzualchemy.test_utilities import initialize_schema


@pytest.fixture(autouse=True)
def global_registry_cleanup():
    """
    CRITICAL: Clean up global registry before EVERY test to prevent access violations.

    This prevents registry pollution that causes Windows fatal exceptions.

    NOTE: clear_registry() already calls gc.collect() internally, so no additional
    garbage collection is needed to prevent double-GC memory corruption.
    """
    # Global registry cleanup before every test
    from kuzualchemy import clear_registry
    clear_registry()  # Already includes gc.collect() internally

    yield

    # Also cleanup after test
    from kuzualchemy import clear_registry
    clear_registry()  # Already includes gc.collect() internally

@pytest.fixture(scope="session")
def test_db_path() -> Generator[Path, None, None]:
    """Create a temporary database path for testing."""
    db_path = Path(tempfile.gettempdir()) / f"test_kuzu_{uuid.uuid4().hex[:8]}"
    yield db_path
    # Cleanup
    if db_path.exists():
        shutil.rmtree(db_path, ignore_errors=True)


@pytest.fixture(scope="function")
def kuzu_connection(test_db_path: Path) -> Generator[KuzuSession, None, None]:
    """Create a Kuzu connection for testing."""
    try:
        conn = KuzuSession(db_path=test_db_path)
        yield conn
    finally:
        conn.close()



@pytest.fixture(scope="function")
def kuzu_session(kuzu_connection: KuzuSession) -> Generator[KuzuSession, None, None]:
    """Create a Kuzu session for testing."""
    session = kuzu_connection
    try:
        yield session
    finally:
        session.close()


# Test model definitions
@kuzu_node("TestUser")
class TestUser(KuzuBaseModel):
    """Test user model."""
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    name: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    email: str = kuzu_field(kuzu_type=KuzuDataType.STRING, unique=True)
    age: int = kuzu_field(kuzu_type=KuzuDataType.INT32, default=0)


@kuzu_node("TestPost")
class TestPost(KuzuBaseModel):
    """Test post model."""
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    title: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    content: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
    author_id: int = kuzu_field(kuzu_type=KuzuDataType.INT64)


@kuzu_relationship("AUTHORED", pairs=[(TestUser, TestPost)])
class AuthoredRelationship(KuzuBaseModel):
    """Test authored relationship."""
    created_at: str = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP, default=KuzuDefaultFunction.CURRENT_TIMESTAMP)
    is_published: bool = kuzu_field(kuzu_type=KuzuDataType.BOOL, default=False)


@kuzu_relationship("FOLLOWS", pairs=[(TestUser, TestUser)])
class FollowsRelationship(KuzuBaseModel):
    """Test follows relationship."""
    followed_at: str = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP, default=KuzuDefaultFunction.CURRENT_TIMESTAMP)
    notification_enabled: bool = kuzu_field(kuzu_type=KuzuDataType.BOOL, default=True)


@pytest.fixture(scope="function")
def test_models() -> Dict[str, Any]:
    """Provide test model classes."""
    return {
        "TestUser": TestUser,
        "TestPost": TestPost,
        "AuthoredRelationship": AuthoredRelationship,
        "FollowsRelationship": FollowsRelationship,
    }


@pytest.fixture(scope="function")
def sample_users() -> List[Dict[str, Any]]:
    """Provide sample user data."""
    return [
        {"id": 1, "name": "Alice Smith", "email": "alice@example.com", "age": 30},
        {"id": 2, "name": "Bob Johnson", "email": "bob@example.com", "age": 25},
        {"id": 3, "name": "Charlie Brown", "email": "charlie@example.com", "age": 35},
    ]


@pytest.fixture(scope="function")
def sample_posts() -> List[Dict[str, Any]]:
    """Provide sample post data."""
    return [
        {"id": 1, "title": "First Post", "content": "Hello World!", "author_id": 1},
        {"id": 2, "title": "Second Post", "content": "Another post", "author_id": 1},
        {"id": 3, "title": "Third Post", "content": "Yet another post", "author_id": 2},
    ]

# conftest or a helpers module
import cProfile
from contextlib import contextmanager
from pathlib import Path

@contextmanager
def profile_to(filename: str, folder: str = "profiles"):
    pr = cProfile.Profile()
    pr.enable()
    try:
        yield
    finally:
        pr.disable()
        Path(folder).mkdir(exist_ok=True)
        pr.dump_stats(str(Path(folder) / f"{filename}.prof"))

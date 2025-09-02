"""
Pytest configuration and shared fixtures for KuzuAlchemy tests.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Generator, Any, Dict, List
from unittest.mock import Mock, patch

import pytest

from kuzualchemy import (
    node,
    relationship,
    KuzuBaseModel,
    Field,
    KuzuDataType,
    foreign_key,
    KuzuSession,
    SessionFactory,
    get_all_ddl,
    get_ddl_for_node,
    get_ddl_for_relationship,
)
from kuzualchemy.constants import KuzuDefaultFunction
from kuzualchemy.test_utilities import initialize_schema


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


@pytest.fixture(scope="function")
def session_factory(test_db_path: Path) -> SessionFactory:
    """Create a session factory for testing."""
    return SessionFactory(test_db_path)


# Test model definitions
@node("TestUser")
class TestUser(KuzuBaseModel):
    """Test user model."""
    id: int = Field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    name: str = Field(kuzu_type=KuzuDataType.STRING, not_null=True)
    email: str = Field(kuzu_type=KuzuDataType.STRING, unique=True)
    age: int = Field(kuzu_type=KuzuDataType.INT32, default=0)


@node("TestPost")
class TestPost(KuzuBaseModel):
    """Test post model."""
    id: int = Field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    title: str = Field(kuzu_type=KuzuDataType.STRING, not_null=True)
    content: str = Field(kuzu_type=KuzuDataType.STRING)
    author_id: int = Field(kuzu_type=KuzuDataType.INT64)


@relationship("AUTHORED", pairs=[(TestUser, TestPost)])
class AuthoredRelationship(KuzuBaseModel):
    """Test authored relationship."""
    created_at: str = Field(kuzu_type=KuzuDataType.TIMESTAMP, default=KuzuDefaultFunction.CURRENT_TIMESTAMP)
    is_published: bool = Field(kuzu_type=KuzuDataType.BOOL, default=False)


@relationship("FOLLOWS", pairs=[(TestUser, TestUser)])
class FollowsRelationship(KuzuBaseModel):
    """Test follows relationship."""
    followed_at: str = Field(kuzu_type=KuzuDataType.TIMESTAMP, default=KuzuDefaultFunction.CURRENT_TIMESTAMP)
    notification_enabled: bool = Field(kuzu_type=KuzuDataType.BOOL, default=True)


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


@pytest.fixture(scope="function")
def initialized_session(kuzu_session: KuzuSession) -> KuzuSession:
    """Create a session with initialized schema."""
    # @@ STEP: Initialize schema using centralized utility
    initialize_schema(kuzu_session)

    return kuzu_session


@pytest.fixture(scope="function")
def populated_session(
    initialized_session: KuzuSession,
    sample_users: List[Dict[str, Any]],
    sample_posts: List[Dict[str, Any]],
    test_models: Dict[str, Any]
) -> KuzuSession:
    """Create a session with sample data."""
    TestUser = test_models["TestUser"]
    TestPost = test_models["TestPost"]

    # Add users
    for user_data in sample_users:
        user = TestUser(**user_data)
        initialized_session.add(user)

    # Add posts
    for post_data in sample_posts:
        post = TestPost(**post_data)
        initialized_session.add(post)

    initialized_session.commit()
    return initialized_session


# Mock fixtures for testing without actual Kuzu installation
@pytest.fixture(scope="function")
def mock_kuzu_connection():
    """Mock Kuzu connection for unit tests."""
    with patch('src.kuzualchemy.kuzu_session.kuzu') as mock_kuzu:
        mock_db = Mock()
        mock_conn = Mock()
        mock_kuzu.Database.return_value = mock_db
        mock_kuzu.Connection.return_value = mock_conn

        # Mock execute method
        mock_conn.execute.return_value = []

        yield mock_conn


@pytest.fixture(scope="function")
def mock_kuzu_session(mock_kuzu_connection):
    """Mock Kuzu session for unit tests."""
    with patch('src.kuzualchemy.kuzu_session.KuzuConnection') as mock_conn_class:
        mock_conn_class.return_value = mock_kuzu_connection
        session = KuzuSession(db_path=":memory:")
        yield session
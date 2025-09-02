"""
Integration tests for KuzuAlchemy ORM.

Tests cover:
- End-to-end workflow scenarios
- Real database operations
- Complex queries with relationships
- Performance and scalability
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from kuzualchemy import (
    node,
    relationship,
    KuzuBaseModel,
    Field,
    KuzuDataType,
    foreign_key,
    KuzuSession,
    get_all_ddl,
)
from kuzualchemy.test_utilities import initialize_schema


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    def setup_method(self):
        """Set up test models and database."""
        @node("IntegrationUser")
        class IntegrationUser(KuzuBaseModel):
            """Integration test user model."""
            id: int = Field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = Field(kuzu_type=KuzuDataType.STRING, not_null=True)
            email: str = Field(kuzu_type=KuzuDataType.STRING, unique=True)
            age: int = Field(kuzu_type=KuzuDataType.INT32, default=0)
            created_at: Optional[str] = Field(kuzu_type=KuzuDataType.STRING, default=None)

        @node("IntegrationPost")
        class IntegrationPost(KuzuBaseModel):
            """Integration test post model."""
            id: int = Field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            title: str = Field(kuzu_type=KuzuDataType.STRING, not_null=True)
            content: str = Field(kuzu_type=KuzuDataType.STRING, default="")
            author_id: int = Field(kuzu_type=KuzuDataType.INT64, foreign_key=foreign_key(IntegrationUser, "id"))
            published: bool = Field(kuzu_type=KuzuDataType.BOOL, default=False)
            created_at: Optional[str] = Field(kuzu_type=KuzuDataType.STRING, default=None)

        @relationship("WROTE", pairs=[(IntegrationUser, IntegrationPost)])
        class Wrote(KuzuBaseModel):
            """Integration test wrote relationship."""
            created_at: datetime = Field(kuzu_type=KuzuDataType.TIMESTAMP, default_factory=datetime.now)
            role: str = Field(kuzu_type=KuzuDataType.STRING, default="author")

        @relationship("FOLLOWS", pairs=[(IntegrationUser, IntegrationUser)])
        class FollowsRel(KuzuBaseModel):
            """Integration test follows relationship."""
            followed_at: Optional[str] = Field(kuzu_type=KuzuDataType.STRING, default=None)
            notification_enabled: bool = Field(kuzu_type=KuzuDataType.BOOL, default=True)

        self.IntegrationUser = IntegrationUser
        self.IntegrationPost = IntegrationPost
        self.Wrote = Wrote
        self.FollowsRel = FollowsRel

        # Create temporary database
        self.temp_db = tempfile.mkdtemp()
        self.db_path = Path(self.temp_db) / "test_integration.db"

    def teardown_method(self):
        """Clean up test database."""
        import shutil
        if Path(self.temp_db).exists():
            shutil.rmtree(self.temp_db, ignore_errors=True)

    def test_complete_crud_workflow(self):
        """Test basic Create, Read, Update, Delete operations."""
        # @@ STEP: Create real database session
        session = KuzuSession(db_path=self.db_path)
        
        # @@ STEP: Initialize schema with DDL
        ddl = get_all_ddl()
        if ddl.strip():
            statements = [stmt.strip() for stmt in ddl.split(';') if stmt.strip()]
            # @@ STEP: Initialize schema using centralized utility
            initialize_schema(session)

        # @@ STEP: Create user
        user = self.IntegrationUser(
            id=1,
            name="Test User",
            email="test@example.com",
            age=25
        )
        session.add(user)
        session.commit()

        # @@ STEP: Verify user exists by querying
        result = session.execute("MATCH (u:IntegrationUser {id: 1}) RETURN u")
        assert result is not None

    def test_complex_query_workflow(self):
        """Test complex query building and execution workflow."""
        # @@ STEP: Create real database session
        session = KuzuSession(db_path=self.db_path)

        # @@ STEP: Initialize schema
        ddl = get_all_ddl()
        if ddl.strip():
            statements = [stmt.strip() for stmt in ddl.split(';') if stmt.strip()]
            # @@ STEP: Initialize schema using centralized utility
            initialize_schema(session)

        # @@ STEP: Insert test data
        users = [
            self.IntegrationUser(id=1, name="Alice Smith", email="alice@example.com", age=30),
            self.IntegrationUser(id=2, name="Bob Johnson", email="bob@example.com", age=25),
            self.IntegrationUser(id=3, name="Charlie Davis", email="charlie@example.com", age=30),
        ]
        for user in users:
            session.add(user)
        session.commit()

        # @@ STEP: Test complex query building
        query = (session.query(self.IntegrationUser)
                .filter_by(age=30)
                .order_by("name")
                .limit(10))

        # @@ STEP: Build and verify query structure
        cypher, params = query.to_cypher()
        assert "MATCH" in cypher
        assert "IntegrationUser" in cypher
        assert params.get(next(iter(params.keys()))) == 30  # age parameter
        
        # @@ STEP: Verify query structure
        assert query._state.limit_value == 10
        assert len(query._state.filters) > 0
        
        # @@ STEP: Execute query and verify results
        results = query.all()
        assert len(results) == 2  # Two users with age 30
        assert results[0].name == "Alice Smith"
        assert results[1].name == "Charlie Davis"

    def test_relationship_workflow(self):
        """Test relationship creation and querying workflow."""
        # @@ STEP: Create real database session
        session = KuzuSession(db_path=self.db_path)

        # @@ STEP: Initialize schema
        ddl = get_all_ddl()
        if ddl.strip():
            statements = [stmt.strip() for stmt in ddl.split(';') if stmt.strip()]
            # @@ STEP: Initialize schema using centralized utility
            initialize_schema(session)

        # @@ STEP: Create users
        user1 = self.IntegrationUser(id=1, name="Alice", email="alice@example.com")
        user2 = self.IntegrationUser(id=2, name="Bob", email="bob@example.com")

        session.add(user1)
        session.add(user2)

        # @@ STEP: Create post
        post = self.IntegrationPost(id=1, title="Test Post", content="Content", author_id=1)
        session.add(post)

        # @@ STEP: Commit and verify
        session.commit()

        # @@ STEP: Query to verify data was inserted
        user_result = list(session.execute("MATCH (u:IntegrationUser {id: 1}) RETURN u.name"))
        assert len(user_result) == 1
        assert user_result[0]['u.name'] == "Alice"
        
        post_result = list(session.execute("MATCH (p:IntegrationPost {id: 1}) RETURN p.title, p.author_id"))
        assert len(post_result) == 1
        assert post_result[0]['p.title'] == "Test Post"
        assert post_result[0]['p.author_id'] == 1

    def test_transaction_workflow(self):
        """Test transaction handling workflow."""
        # @@ STEP: Create real database session
        session = KuzuSession(db_path=self.db_path, autocommit=False)

        # @@ STEP: Initialize schema
        ddl = get_all_ddl()
        if ddl.strip():
            statements = [stmt.strip() for stmt in ddl.split(';') if stmt.strip()]
            # @@ STEP: Initialize schema using centralized utility
            initialize_schema(session)

        # @@ STEP: Test transaction with context manager
        with session.begin():
            # @@ STEP: Create user
            user = self.IntegrationUser(id=1, name="Alice", email="alice@example.com")
            session.add(user)

            # @@ STEP: Create post
            post = self.IntegrationPost(id=1, title="Test Post", content="Content", author_id=1)
            session.add(post)
            # Transaction commits automatically when exiting context

        # @@ STEP: Verify data was committed
        result = list(session.execute("MATCH (u:IntegrationUser {id: 1}) RETURN u.name"))
        assert len(result) == 1
        assert result[0]['u.name'] == "Alice"
        
        # @@ STEP: Test rollback scenario
        session.begin()
        user2 = self.IntegrationUser(id=2, name="Bob", email="bob@example.com")
        session.add(user2)
        session.rollback()
        
        # @@ STEP: Verify rollback worked
        result = list(session.execute("MATCH (u:IntegrationUser {id: 2}) RETURN u"))
        assert len(result) == 0

    def test_error_handling_workflow(self):
        """Test error handling in real scenarios."""
        # @@ STEP: Create real database session
        session = KuzuSession(db_path=self.db_path)

        # @@ STEP: Initialize schema
        ddl = get_all_ddl()
        if ddl.strip():
            statements = [stmt.strip() for stmt in ddl.split(';') if stmt.strip()]
            # @@ STEP: Initialize schema using centralized utility
            initialize_schema(session)

        # @@ STEP: Insert a user
        user = self.IntegrationUser(id=1, name="Alice", email="alice@example.com")
        session.add(user)
        session.commit()

        # @@ STEP: Try to insert duplicate user with same primary key
        duplicate_user = self.IntegrationUser(id=1, name="Bob", email="bob@example.com")
        session.add(duplicate_user)
        
        # @@ STEP: This should fail due to primary key constraint
        try:
            session.commit()
            assert False, "Should have raised an error for duplicate primary key"
        except RuntimeError as e:
            # Error should be propagated, not swallowed
            assert "1" in str(e) or "primary" in str(e).lower() or "constraint" in str(e).lower()


class TestPerformanceScenarios:
    """Test performance and scalability scenarios."""

    def setup_method(self):
        """Set up performance test models."""
        @node("PerfUser")
        class PerfUser(KuzuBaseModel):
            """Performance test user model."""
            id: int = Field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = Field(kuzu_type=KuzuDataType.STRING, index=True)
            email: str = Field(kuzu_type=KuzuDataType.STRING, unique=True)
            age: int = Field(kuzu_type=KuzuDataType.INT32, default=0)

        @relationship("PERF_FOLLOWS", pairs=[(PerfUser, PerfUser)])
        class PerfFollows(KuzuBaseModel):
            """Performance test follows relationship."""
            since: datetime = Field(kuzu_type=KuzuDataType.TIMESTAMP, default_factory=datetime.now)
            strength: float = Field(kuzu_type=KuzuDataType.DOUBLE, default=1.0)

        self.PerfUser = PerfUser
        self.PerfFollows = PerfFollows

        # Create temporary database
        self.temp_db = tempfile.mkdtemp()
        self.db_path = Path(self.temp_db) / "test_performance.db"

    def teardown_method(self):
        """Clean up test database."""
        import shutil
        if Path(self.temp_db).exists():
            shutil.rmtree(self.temp_db, ignore_errors=True)

    def test_bulk_operations(self):
        """Test bulk insert operations."""
        # @@ STEP: Create real database session
        session = KuzuSession(db_path=self.db_path)

        # @@ STEP: Initialize schema
        ddl = get_all_ddl()
        if ddl.strip():
            statements = [stmt.strip() for stmt in ddl.split(';') if stmt.strip()]
            # @@ STEP: Initialize schema using centralized utility
            initialize_schema(session)

        # @@ STEP: Create many users
        users = []
        for i in range(100):
            user = self.PerfUser(
                id=i,
                name=f"User {i}",
                email=f"user{i}@example.com",
                age=20 + (i % 50)  # Vary ages for testing
            )
            users.append(user)
            session.add(user)

        # @@ STEP: Commit all at once
        session.commit()

        # @@ STEP: Verify bulk operations
        result = list(session.execute("MATCH (u:PerfUser) RETURN count(u) as count"))
        assert result[0]['count'] == 100
        
        # @@ STEP: Test query on bulk data
        age_30_result = list(session.execute("MATCH (u:PerfUser) WHERE u.age = 30 RETURN count(u) as count"))
        assert age_30_result[0]['count'] == 2  # Users with id 10 and 60

    def test_complex_query_performance(self):
        """Test complex query building performance."""
        # @@ STEP: Create real database session
        session = KuzuSession(db_path=self.db_path)
        
        # @@ STEP: Initialize schema
        ddl = get_all_ddl()
        if ddl.strip():
            statements = [stmt.strip() for stmt in ddl.split(';') if stmt.strip()]
            # @@ STEP: Initialize schema using centralized utility
            initialize_schema(session)

        # @@ STEP: Insert test data
        for i in range(50):
            user = self.PerfUser(
                id=i,
                name=f"User {str(i).zfill(3)}",  # Pad for consistent sorting
                email=f"user{i}@example.com",
                age=20 + (i % 10)  # Ages from 20-29
            )
            session.add(user)
        session.commit()
        
        # @@ STEP: Build complex query
        query = (session.query(self.PerfUser)
                .filter_by(age=25)
                .order_by("name")
                .offset(2)
                .limit(3))

        # @@ STEP: Verify query builds correctly
        cypher, params = query.to_cypher()
        assert "MATCH" in cypher
        assert "PerfUser" in cypher
        assert "ORDER BY" in cypher
        assert "SKIP" in cypher  # Kuzu uses SKIP instead of OFFSET
        assert "LIMIT" in cypher
        
        # @@ STEP: Execute and verify results
        results = query.all()
        assert len(results) <= 3  # Limited to 3
        # All results should have age 25
        for user in results:
            assert user.age == 25
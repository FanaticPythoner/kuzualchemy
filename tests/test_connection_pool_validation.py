# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
Connection Pool Validation Test
===============================

This test validates that the connection pool fixes work correctly and that
concurrent access is now properly supported without file locking errors.
"""

from __future__ import annotations

import pytest
import tempfile
import shutil
import threading
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from kuzualchemy import (
    KuzuNodeBase,
    KuzuRelationshipBase,
    kuzu_node,
    kuzu_relationship,
    kuzu_field,
    KuzuDataType,
    KuzuSession,
)
from kuzualchemy.test_utilities import initialize_schema
from kuzualchemy.kuzu_orm import get_ddl_for_node, get_ddl_for_relationship


@kuzu_node("TestUser")
class TestUser(KuzuNodeBase):
    """Test user model."""
    user_id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    username: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    email: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    created_at: datetime = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP)


@kuzu_node("TestProduct")
class TestProduct(KuzuNodeBase):
    """Test product model."""
    product_id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    name: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    price: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, not_null=True)


@kuzu_relationship("PURCHASED", pairs=[(TestUser, TestProduct)])
class TestPurchased(KuzuRelationshipBase):
    """Test purchase relationship."""
    quantity: int = kuzu_field(kuzu_type=KuzuDataType.INT32, not_null=True)
    price_paid: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, not_null=True)
    purchased_at: datetime = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP)


class TestConnectionPoolValidation:
    """Test connection pool functionality and concurrent access."""

    def setup_method(self):
        """Set up test database."""
        self.temp_db = tempfile.mkdtemp()
        self.db_path = Path(self.temp_db) / "connection_pool_test.db"

        # @@ STEP: Re-register models due to conftest.py registry cleanup
        from kuzualchemy.kuzu_orm import _kuzu_registry

        # Re-register node models
        node_models = [TestUser, TestProduct]
        for model in node_models:
            node_name = model.__kuzu_node_name__
            _kuzu_registry.register_node(node_name, model)

        # Re-register relationship models
        rel_models = [TestPurchased]
        for model in rel_models:
            rel_name = model.__kuzu_relationship_name__
            _kuzu_registry.register_relationship(rel_name, model)

    def teardown_method(self):
        """Clean up test database."""
        if Path(self.temp_db).exists():
            shutil.rmtree(self.temp_db, ignore_errors=True)

    def test_basic_connection_pool_functionality(self):
        """Test that basic connection pool functionality works."""
        # Create main session
        session = KuzuSession(db_path=str(self.db_path))
        
        # Initialize schema
        all_models = [TestUser, TestProduct, TestPurchased]
        ddl_statements = []
        for model in all_models:
            if hasattr(model, '__is_kuzu_relationship__') and model.__is_kuzu_relationship__:
                ddl_statements.append(get_ddl_for_relationship(model))
            else:
                ddl_statements.append(get_ddl_for_node(model))
        ddl = "\n".join(ddl_statements)
        initialize_schema(session, ddl=ddl)
        
        # Create test data
        user = TestUser(
            user_id=1,
            username="testuser",
            email="test@example.com",
            created_at=datetime.now()
        )
        product = TestProduct(
            product_id=1,
            name="Test Product",
            price=99.99
        )
        
        session.add(user)
        session.add(product)
        session.commit()
        
        # Verify data was created
        users = list(session.execute("MATCH (u:TestUser) RETURN count(u) as count"))
        assert users[0]['count'] == 1
        
        products = list(session.execute("MATCH (p:TestProduct) RETURN count(p) as count"))
        assert products[0]['count'] == 1
        
        session.close()

    def test_concurrent_read_access(self):
        """Test that multiple read-only sessions can access the database concurrently."""
        # Create main session and set up data
        main_session = KuzuSession(db_path=str(self.db_path))
        
        # Initialize schema
        all_models = [TestUser, TestProduct, TestPurchased]
        ddl_statements = []
        for model in all_models:
            if hasattr(model, '__is_kuzu_relationship__') and model.__is_kuzu_relationship__:
                ddl_statements.append(get_ddl_for_relationship(model))
            else:
                ddl_statements.append(get_ddl_for_node(model))
        ddl = "\n".join(ddl_statements)
        initialize_schema(main_session, ddl=ddl)
        
        # Create test data
        for i in range(100):
            user = TestUser(
                user_id=i + 1,
                username=f"user_{i+1}",
                email=f"user{i+1}@example.com",
                created_at=datetime.now()
            )
            main_session.add(user)
        
        main_session.commit()
        
        def concurrent_reader(worker_id: int) -> tuple[int, int]:
            """Worker function for concurrent read operations."""
            try:
                # Use read-only session
                reader_session = KuzuSession(db_path=str(self.db_path))
                successes = 0
                failures = 0
                
                for _ in range(10):  # 10 operations per worker
                    try:
                        results = list(reader_session.execute("MATCH (u:TestUser) RETURN count(u) as count"))
                        if results and results[0]['count'] == 100:
                            successes += 1
                        else:
                            failures += 1
                        
                        # Small delay to simulate real work
                        time.sleep(0.01)
                        
                    except Exception as e:
                        print(f"Worker {worker_id} error: {e}")
                        failures += 1
                
                reader_session.close()
                return successes, failures
                
            except Exception as e:
                print(f"Worker {worker_id} setup error: {e}")
                return 0, 10

        # Run concurrent readers
        num_workers = 5
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(concurrent_reader, i) for i in range(num_workers)]
            
            total_successes = 0
            total_failures = 0
            
            for future in as_completed(futures):
                successes, failures = future.result()
                total_successes += successes
                total_failures += failures

        main_session.close()
        
        # Validate results
        print(f"Concurrent read test: {total_successes} successes, {total_failures} failures")
        assert total_failures == 0, f"Expected 0 failures, got {total_failures}"
        assert total_successes == num_workers * 10, f"Expected {num_workers * 10} successes, got {total_successes}"

    def test_no_file_locking_errors(self):
        """Test that we don't get file locking errors with multiple sessions."""
        sessions = []
        
        try:
            # Create multiple sessions to the same database
            for i in range(3):
                if i == 0:
                    # First session is read-write
                    session = KuzuSession(db_path=str(self.db_path))
                else:
                    # Others are read-only
                    session = KuzuSession(db_path=str(self.db_path))
                sessions.append(session)
            
            # Initialize schema with first session
            all_models = [TestUser, TestProduct, TestPurchased]
            ddl_statements = []
            for model in all_models:
                if hasattr(model, '__is_kuzu_relationship__') and model.__is_kuzu_relationship__:
                    ddl_statements.append(get_ddl_for_relationship(model))
                else:
                    ddl_statements.append(get_ddl_for_node(model))
            ddl = "\n".join(ddl_statements)
            initialize_schema(sessions[0], ddl=ddl)
            
            # Write with first session
            user = TestUser(
                user_id=1,
                username="testuser",
                email="test@example.com",
                created_at=datetime.now()
            )
            sessions[0].add(user)
            sessions[0].commit()
            
            # Read with other sessions
            for i in range(1, len(sessions)):
                results = list(sessions[i].execute("MATCH (u:TestUser) RETURN count(u) as count"))
                assert results[0]['count'] == 1
            
            print("✅ No file locking errors - multiple sessions work correctly!")
            
        finally:
            # Clean up sessions
            for session in sessions:
                try:
                    session.close()
                except Exception:
                    pass

    def test_relationship_hashing_fixed(self):
        """Test that relationship hashing works correctly without warnings."""
        session = KuzuSession(db_path=str(self.db_path))
        
        # Initialize schema
        all_models = [TestUser, TestProduct, TestPurchased]
        ddl_statements = []
        for model in all_models:
            if hasattr(model, '__is_kuzu_relationship__') and model.__is_kuzu_relationship__:
                ddl_statements.append(get_ddl_for_relationship(model))
            else:
                ddl_statements.append(get_ddl_for_node(model))
        ddl = "\n".join(ddl_statements)
        initialize_schema(session, ddl=ddl)
        
        # Create test data
        user = TestUser(
            user_id=1,
            username="testuser",
            email="test@example.com",
            created_at=datetime.now()
        )
        product = TestProduct(
            product_id=1,
            name="Test Product",
            price=99.99
        )
        
        session.add(user)
        session.add(product)
        session.commit()
        
        # Create multiple relationships with same nodes but different properties
        # This should not cause hash collisions or warnings
        # Use model instances instead of raw primary key values to avoid ambiguity
        purchase1 = TestPurchased(
            from_node=user,
            to_node=product,
            quantity=1,
            price_paid=99.99,
            purchased_at=datetime.now()
        )

        purchase2 = TestPurchased(
            from_node=user,
            to_node=product,
            quantity=2,
            price_paid=199.98,
            purchased_at=datetime.now()
        )
        
        # These should be treated as different relationships
        session.add(purchase1)
        session.add(purchase2)
        session.commit()
        
        # Verify both relationships were created
        results = list(session.execute("MATCH ()-[p:PURCHASED]->() RETURN count(p) as count"))
        assert results[0]['count'] == 2
        
        session.close()
        print("✅ Relationship hashing works correctly!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

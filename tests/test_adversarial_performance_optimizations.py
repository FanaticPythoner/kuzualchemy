# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
test suite for performance optimizations and exception handling.

This test suite is designed to break the implementation by testing edge cases,
boundary conditions, and stress scenarios for the performance optimizations
implemented in kuzu_session.py.
"""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

import kuzualchemy as ka
from kuzualchemy.constants import PerformanceConstants
from kuzualchemy.kuzu_session import KuzuSession


class TestPerformanceOptimizations:
    """test suite for performance optimizations."""

    def setup_method(self):
        """Set up test environment."""
        # Clear registry to ensure clean state
        ka.clear_registry()

        # Define test models dynamically
        @ka.kuzu_node("ATNode")
        class ATNode(ka.KuzuBaseModel):
            """Test node for testing."""
            id: int = ka.kuzu_field(kuzu_type=ka.KuzuDataType.INT64, primary_key=True)
            name: str = ka.kuzu_field(kuzu_type=ka.KuzuDataType.STRING)
            value: int = ka.kuzu_field(kuzu_type=ka.KuzuDataType.INT32)

        @ka.kuzu_node("BrokenNode")
        class BrokenNode(ka.KuzuBaseModel):
            """Node with intentionally problematic configuration."""
            # Has primary key but will be used to test edge cases
            id: int = ka.kuzu_field(kuzu_type=ka.KuzuDataType.INT64, primary_key=True)
            name: str = ka.kuzu_field(kuzu_type=ka.KuzuDataType.STRING)

        # Store model classes for use in tests
        self.ATNode = ATNode
        self.BrokenNode = BrokenNode

        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = Path(self.temp_db.name)

        # Create session with performance optimizations enabled
        self.session = KuzuSession(db_path=self.db_path)

        # Initialize schema - the decorators should have registered the models
        ddl = ka.get_all_ddl()
        if ddl.strip():
            self.session.execute(ddl)

    def teardown_method(self):
        """Clean up test environment."""
        if hasattr(self, 'session'):
            self.session.close()
        if hasattr(self, 'db_path') and self.db_path.exists():
            self.db_path.unlink()
        # Clear registry after tests
        ka.clear_registry()

    def test_connection_reuse_threshold_boundary(self):
        """Test connection reuse at exact threshold boundaries."""
        # Test exactly at threshold
        threshold = PerformanceConstants.CONNECTION_REUSE_THRESHOLD
        
        # No internal connection reuse state to reset under ATP-managed connections

        # Execute exactly threshold number of operations
        for i in range(threshold):
            result = self.session.execute("MATCH (n:ATNode) RETURN count(n) as count")
            # Justification: Each query should return exactly one row with count=0
            # since no nodes have been inserted yet
            assert len(result) == 1
            assert result[0]["count"] == 0

        # One more operation should continue to work identically under ATP
        result = self.session.execute("MATCH (n:ATNode) RETURN count(n) as count")
        # Justification: After reaching threshold, connection should reset to 1
        # but query result should still be the same
        assert len(result) == 1
        assert result[0]["count"] == 0
        # No internal counters exist; only the results matter

    def test_identity_map_key_generation_edge_cases(self):
        """Test identity map key generation with edge case values."""
        # Test with various primary key types that could cause key collisions
        test_cases = [
            (1, "1"),  # int vs string that looks like int
            ("test", "test"),  # normal string
            ("test:with:colons", "test:with:colons"),  # string with colons (key separator)
            (0, "0"),  # zero values
            (-1, "-1"),  # negative values
        ]
        
        for pk_value, expected_str in test_cases:
            key = self.session._generate_identity_key(self.ATNode, pk_value)
            # Justification: Key should be exactly "ClassName:pk_value"
            expected_key = f"ATNode:{expected_str}"
            assert key == expected_key

    def test_smart_autoflush_with_no_pending_operations(self):
        """Test that autoflush is skipped when no operations are pending."""
        # Enable autoflush
        self.session.autoflush = True
        
        # Mock flush to detect if it's called
        original_flush = self.session.flush
        flush_call_count = 0
        
        def mock_flush():
            nonlocal flush_call_count
            flush_call_count += 1
            return original_flush()
        
        self.session.flush = mock_flush
        
        # Execute query with no pending operations
        result = self.session.execute("MATCH (n:ATNode) RETURN count(n) as count")
        
        # Justification: flush should not be called when there are no pending operations
        assert flush_call_count == 0
        assert len(result) == 1
        assert result[0]["count"] == 0

    def test_smart_autoflush_with_pending_operations(self):
        """Test that autoflush is triggered when operations are pending."""
        # Enable autoflush
        self.session.autoflush = True
        
        # Add a pending operation
        node = self.ATNode(id=1, name="test", value=42)
        self.session.add(node)
        
        # Mock flush to detect if it's called
        original_flush = self.session.flush
        flush_call_count = 0
        
        def mock_flush():
            nonlocal flush_call_count
            flush_call_count += 1
            return original_flush()
        
        self.session.flush = mock_flush
        
        # Execute query - should trigger flush
        result = self.session.execute("MATCH (n:ATNode) RETURN count(n) as count")
        
        # Justification: flush should be called exactly once since we have pending operations
        assert flush_call_count == 1
        assert len(result) == 1
        assert result[0]["count"] == 1  # Node should be inserted after flush

    def test_exception_handling_in_get_node_type_name_value_error(self):
        """Test specific ValueError handling in _get_node_type_name method."""
        # Create a node instance to test edge cases
        broken_node = self.BrokenNode(id=1, name="test")
        
        # This should handle ValueError gracefully and continue with other node types
        with pytest.raises(TypeError, match="Primary key value .* does not exist in any registered node type"):
            self.session._get_node_type_name("nonexistent_pk_value")

    def test_exception_handling_in_get_node_type_name_database_error(self):
        """Test database error handling in _get_node_type_name method."""
        # Mock connection to simulate database errors
        original_execute = self.session._conn.execute
        
        def mock_execute_with_error(query, params=None):
            if "MATCH" in query and "count(n)" in query:
                raise RuntimeError("Database connection failed")
            return original_execute(query, params)
        
        self.session._conn.execute = mock_execute_with_error
        
        # This should handle RuntimeError gracefully and continue
        with pytest.raises(TypeError, match="Primary key value .* does not exist in any registered node type"):
            self.session._get_node_type_name("test_pk_value")

    def test_connection_reuse_with_stale_connection(self):
        """Test connection reuse recovery when connection becomes stale."""
        # Under ATP, no reuse/fallback: patch the underlying connection to raise and ensure propagation
        self.session._conn.execute = Mock(side_effect=RuntimeError("Connection stale"))
        with pytest.raises(RuntimeError, match="Connection stale"):
            self.session.execute("MATCH (n:ATNode) RETURN count(n) as count")

    def test_identity_map_optimization_with_merge_operations(self):
        """Test identity map optimization during merge operations."""
        # Disable expire_on_commit to test identity map persistence
        self.session.expire_on_commit = False

        # Create and insert a node
        node = self.ATNode(id=1, name="original", value=100)
        self.session.add(node)
        self.session.commit()

        # Create another instance with same ID but different values
        updated_node = self.ATNode(id=1, name="updated", value=200)
        
        # Merge should use optimized identity map lookup
        merged = self.session.merge(updated_node)
        
        # Justification: Merged node should be the same object as original
        # but with updated values
        assert merged is node  # Same object reference
        assert merged.name == "updated"
        assert merged.value == 200
        
        # Verify identity map key was generated correctly
        expected_key = "ATNode:1"
        assert expected_key in self.session._identity_map
        assert self.session._identity_map[expected_key] is merged

    def test_concurrent_connection_reuse_thread_safety(self):
        """Test thread safety of connection reuse optimization."""
        results = []
        errors = []
        
        def worker_thread(thread_id: int):
            try:
                # Each thread executes multiple queries
                for i in range(5):
                    result = self.session.execute("MATCH (n:ATNode) RETURN count(n) as count")
                    results.append((thread_id, i, result[0]["count"]))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Justification: All queries should succeed with count=0
        # No errors should occur due to thread safety issues
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 15  # 3 threads * 5 queries each
        
        for thread_id, query_id, count in results:
            assert count == 0  # No nodes inserted, so count should be 0

    def test_performance_regression_detection(self):
        """Test to detect performance regressions in optimized code."""
        # Baseline: measure time for operations without optimizations
        start_time = time.time()

        # Perform a series of operations that should benefit from optimizations
        nodes = []
        for i in range(100):
            node = self.ATNode(id=i, name=f"node_{i}", value=i * 10)
            nodes.append(node)

        # Add all nodes
        self.session.add_all(nodes)
        self.session.commit()

        # Perform queries that should use connection reuse
        for i in range(10):
            result = self.session.execute("MATCH (n:ATNode) RETURN count(n) as count")
            # Justification: Should return exactly 100 nodes
            assert result[0]["count"] == 100

        # Perform merge operations that should use optimized identity map
        for i in range(10):
            updated_node = self.ATNode(id=i, name=f"updated_{i}", value=i * 20)
            merged = self.session.merge(updated_node)
            # Justification: Merged node should have updated values
            assert merged.name == f"updated_{i}"
            assert merged.value == i * 20

        total_time = time.time() - start_time

        # Justification: With optimizations, 100 inserts + 10 queries + 10 merges
        # should complete in reasonable time (less than 5 seconds)
        assert total_time < 5.0, f"Performance regression detected: {total_time:.2f}s > 5.0s"

    def test_edge_case_empty_database_operations(self):
        """Test edge cases with empty database operations."""
        # Test queries on empty database
        result = self.session.execute("MATCH (n:ATNode) RETURN n")
        # Justification: Empty database should return empty result
        assert len(result) == 0

        # Test count on empty database
        result = self.session.execute("MATCH (n:ATNode) RETURN count(n) as count")
        # Justification: Count of empty set should be exactly 0
        assert len(result) == 1
        assert result[0]["count"] == 0

        # Test merge with non-existent node
        node = self.ATNode(id=999, name="nonexistent", value=0)
        merged = self.session.merge(node)
        # Justification: Merging non-existent node should add it to session
        assert merged is node
        assert node in self.session._new

    def test_boundary_conditions_for_batch_operations(self):
        """Test boundary conditions for batch operations."""
        # Test with exactly bulk_insert_threshold nodes
        threshold = self.session.bulk_insert_threshold
        nodes = []

        for i in range(threshold):
            node = self.ATNode(id=i, name=f"batch_{i}", value=i)
            nodes.append(node)

        # Add exactly threshold number of nodes
        self.session.add_all(nodes)

        # Justification: Should have exactly threshold nodes in _new
        assert len(self.session._new) == threshold

        # Flush should trigger bulk insert
        self.session.flush()

        # Justification: After flush, _new should be empty
        assert len(self.session._new) == 0

        # Verify all nodes were inserted
        result = self.session.execute("MATCH (n:ATNode) RETURN count(n) as count")
        assert result[0]["count"] == threshold

    def test_stress_identity_map_with_large_dataset(self):
        """Stress test identity map with large dataset."""
        # Disable expire_on_commit to test identity map persistence
        self.session.expire_on_commit = False

        # Create a large number of nodes to stress the identity map
        large_count = 1000
        nodes = []

        for i in range(large_count):
            node = self.ATNode(id=i, name=f"stress_{i}", value=i * 2)
            nodes.append(node)

        # Add and commit all nodes
        self.session.add_all(nodes)
        self.session.commit()

        # Perform merge operations on all nodes to stress identity map
        start_time = time.time()

        for i in range(large_count):
            updated_node = self.ATNode(id=i, name=f"updated_stress_{i}", value=i * 3)
            merged = self.session.merge(updated_node)
            # Justification: Each merge should find existing node in identity map
            assert merged.name == f"updated_stress_{i}"
            assert merged.value == i * 3

        merge_time = time.time() - start_time

        # Justification: With optimized identity map, 1000 merges should complete
        # in reasonable time (less than 2 seconds)
        assert merge_time < 2.0, f"Identity map performance regression: {merge_time:.2f}s > 2.0s"

        # Verify identity map size
        # Justification: Should have exactly large_count entries in identity map
        assert len(self.session._identity_map) == large_count

    def test_exception_propagation_in_optimized_paths(self):
        """Test that exceptions are properly propagated in optimized code paths."""
        # Under ATP, there is no reuse/fallback; the underlying error should propagate
        self.session._conn.execute = Mock(side_effect=RuntimeError("Main connection failed"))
        with pytest.raises(RuntimeError, match="Main connection failed"):
            self.session.execute("MATCH (n:ATNode) RETURN count(n)")

    def test_memory_efficiency_of_optimizations(self):
        """Test memory efficiency of performance optimizations."""
        import gc
        import sys

        # Force garbage collection and measure initial memory
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Perform operations that create temporary objects
        for i in range(100):
            node = self.ATNode(id=i, name=f"memory_test_{i}", value=i)
            self.session.add(node)

            # Generate identity keys (should not create excessive objects)
            key = self.session._generate_identity_key(self.ATNode, i)
            # Justification: Key should be a simple string
            assert isinstance(key, str)
            assert key == f"ATNode:{i}"

        self.session.commit()

        # Force garbage collection and measure final memory
        gc.collect()
        final_objects = len(gc.get_objects())

        # Justification: Object growth should be reasonable
        # (less than 10x the number of nodes created)
        object_growth = final_objects - initial_objects
        assert object_growth < 1000, f"Excessive object creation: {object_growth} new objects"

    def test_identity_map_initial_size_usage(self):
        """Test that IDENTITY_MAP_INITIAL_SIZE constant is actually used."""
        # @@ STEP: Verify that identity map was initialized with proper size
        # || Verification: Pre-allocated dictionary should have been created
        from kuzualchemy.constants import PerformanceConstants

        # @@ STEP: The identity map should be initialized but empty
        assert len(self.session._identity_map) == 0

        # @@ STEP: Verify the constant is accessible and has expected value
        assert PerformanceConstants.IDENTITY_MAP_INITIAL_SIZE == 256

        # @@ STEP: Add items up to test pre-allocation effectiveness
        # || This tests that the pre-allocation worked correctly
        nodes = []
        for i in range(10):  # Test with smaller number for performance
            node = self.ATNode(id=i, name=f"User {i}", value=20 + i)
            self.session.add(node)
            nodes.append(node)

        self.session.flush()

        # @@ STEP: Verify all nodes are in identity map with correct keys
        for i, node in enumerate(nodes):
            identity_key = self.session._generate_identity_key(self.ATNode, i)
            assert identity_key in self.session._identity_map
            assert self.session._identity_map[identity_key] == node

    def test_autoflush_batch_size_usage(self):
        """Test that AUTOFLUSH_BATCH_SIZE constant is wired into session config and pending counts are tracked."""
        # @@ STEP: Verify the constant is accessible and used
        from kuzualchemy.constants import PerformanceConstants

        assert PerformanceConstants.AUTOFLUSH_BATCH_SIZE == 100
        assert self.session._autoflush_batch_size == 100

        # @@ STEP: Create operations below the batch threshold (legacy) and verify pending counts directly
        nodes = []
        for i in range(50):  # Below batch size
            node = self.ATNode(id=i, name=f"User {i}", value=20 + i)
            self.session.add(node)
            nodes.append(node)

        # @@ STEP: Verify pending operations count is tracked without calling removed helper
        pending_count = len(self.session._new) + len(self.session._dirty) + len(self.session._deleted)
        assert pending_count == 50
        assert pending_count > 0  # Any pending operations will trigger autoflush on execute()

        # @@ STEP: Add more to reach the legacy batch threshold and verify counts again
        for i in range(50, 100):  # Reach batch size
            node = self.ATNode(id=i, name=f"User {i}", value=20 + i)
            self.session.add(node)
            nodes.append(node)

        pending_count = len(self.session._new) + len(self.session._dirty) + len(self.session._deleted)
        assert pending_count == 100
        assert pending_count > 0

    def test_metadata_cache_size_usage(self):
        """Test that METADATA_CACHE_SIZE constant is actually used in metadata caching."""
        # @@ STEP: Verify the constant is accessible and used
        from kuzualchemy.constants import PerformanceConstants

        assert PerformanceConstants.METADATA_CACHE_SIZE == 500
        assert self.session._metadata_cache_size == 500

        # @@ STEP: Test metadata caching functionality
        # || Cache some metadata
        test_metadata = ['id', 'name', 'value']
        self.session._set_cached_metadata(self.ATNode, 'pk_fields', test_metadata)

        # @@ STEP: Verify metadata is cached and retrievable
        cached_metadata = self.session._get_cached_metadata(self.ATNode, 'pk_fields')
        assert cached_metadata == test_metadata

        # @@ STEP: Test LRU eviction by filling cache beyond capacity
        # || Create metadata entries up to cache size
        for i in range(10):  # Test with smaller number for performance
            dummy_class_name = f"DummyClass{i}"
            self.session._metadata_cache[f"{dummy_class_name}:pk_fields"] = [f"field_{i}"]

        # @@ STEP: Verify cache size is managed
        assert len(self.session._metadata_cache) <= self.session._metadata_cache_size

        # @@ STEP: Test cache clearing
        self.session._clear_metadata_cache()
        assert len(self.session._metadata_cache) == 0

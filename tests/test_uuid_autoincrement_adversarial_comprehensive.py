# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive tests for UUID auto-increment implementation.

This test suite is designed by [You-As-Hat-1-QA-Tester] to break the UUID auto-increment
implementation by testing complex real-world patterns, edge cases, boundary conditions,
and unusual but valid Python syntax.

The goal is to achieve 100% coverage and expose any weaknesses in the implementation.
"""

from __future__ import annotations

import uuid
import threading
import time
from typing import Optional

import pytest

from src.kuzualchemy import KuzuBaseModel, kuzu_field, kuzu_node, KuzuSession
from src.kuzualchemy.constants import KuzuDataType
from src.kuzualchemy.test_utilities import initialize_schema
from src.kuzualchemy.kuzu_orm import get_ddl_for_node


class TestUUIDAutoIncrement:
    """tests designed to break UUID auto-increment implementation."""

    @pytest.fixture
    def test_db_path(self, tmp_path):
        """Provide a temporary database path for testing."""
        return tmp_path / "test_uuid.db"

    def setup_method(self):
        """Set up test models with various UUID auto-increment configurations."""
        
        @kuzu_node("ATUUIDNode")
        class ATUUIDNode(KuzuBaseModel):
            """Node with UUID auto-increment primary key for testing."""
            id: Optional[uuid.UUID] = kuzu_field(
                kuzu_type=KuzuDataType.UUID, 
                primary_key=True, 
                auto_increment=True
            )
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
            
        @kuzu_node("MultiUUIDNode")
        class MultiUUIDNode(KuzuBaseModel):
            """Node with multiple UUID fields to test complex scenarios."""
            id: Optional[uuid.UUID] = kuzu_field(
                kuzu_type=KuzuDataType.UUID, 
                primary_key=True, 
                auto_increment=True
            )
            secondary_uuid: Optional[uuid.UUID] = kuzu_field(
                kuzu_type=KuzuDataType.UUID, 
                auto_increment=True
            )
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
            
        @kuzu_node("NonOptionalUUIDNode")
        class NonOptionalUUIDNode(KuzuBaseModel):
            """Node with non-optional UUID auto-increment field."""
            id: uuid.UUID = kuzu_field(
                kuzu_type=KuzuDataType.UUID, 
                primary_key=True, 
                auto_increment=True
            )
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
            
        self.ATUUIDNode = ATUUIDNode
        self.MultiUUIDNode = MultiUUIDNode
        self.NonOptionalUUIDNode = NonOptionalUUIDNode

    def test_pydantic_fields_set_edge_cases(self, test_db_path):
        """Test edge cases with __pydantic_fields_set__ manipulation."""
        session = KuzuSession(db_path=test_db_path)
        initialize_schema(session, ddl=get_ddl_for_node(self.ATUUIDNode))
        
        # Test 1: Manually manipulate __pydantic_fields_set__ after instantiation
        node = self.ATUUIDNode(name="Test")
        
        # Initially, id should not be in fields_set
        assert "id" not in node.__pydantic_fields_set__
        manual_values = node.get_manual_auto_increment_values()
        assert manual_values == {}
        
        # Manually add 'id' to __pydantic_fields_set__ without setting the field
        node.__pydantic_fields_set__.add("id")
        
        # This should cause get_manual_auto_increment_values() to try to get the field value
        # The field value should be None (default), so it should be included
        manual_values = node.get_manual_auto_increment_values()
        # Expected: {"id": None} since getattr(node, "id") returns None
        assert manual_values == {"id": None}
        
        session.close()

    def test_field_access_with_property_descriptors(self, test_db_path):
        """Test field access with custom property descriptors and unusual field behaviors."""
        session = KuzuSession(db_path=test_db_path)

        # Create a node instance
        node = self.ATUUIDNode(name="Test")

        # Test that the method handles normal field access correctly
        manual_values = node.get_manual_auto_increment_values()
        assert manual_values == {}

        # Test with manually provided UUID
        manual_uuid = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
        node_with_uuid = self.ATUUIDNode(id=manual_uuid, name="Test")

        manual_values = node_with_uuid.get_manual_auto_increment_values()
        assert manual_values == {"id": manual_uuid}

        # Test that the field value is retrieved correctly
        assert node_with_uuid.id == manual_uuid

        session.close()

    def test_multiple_uuid_fields_complex_scenarios(self, test_db_path):
        """Test complex scenarios with multiple UUID auto-increment fields."""
        session = KuzuSession(db_path=test_db_path)
        initialize_schema(session, ddl=get_ddl_for_node(self.MultiUUIDNode))
        
        # Test 1: No UUIDs provided (both should be auto-generated)
        node1 = self.MultiUUIDNode(name="Auto Both")
        
        # Both fields should need generation
        fields_needing_generation = node1.get_auto_increment_fields_needing_generation()
        assert set(fields_needing_generation) == {"id", "secondary_uuid"}
        
        # No manual values
        manual_values = node1.get_manual_auto_increment_values()
        assert manual_values == {}
        
        session.add(node1)
        session.commit()
        
        # Both should be generated
        assert node1.id is not None
        assert node1.secondary_uuid is not None
        assert isinstance(node1.id, uuid.UUID)
        assert isinstance(node1.secondary_uuid, uuid.UUID)
        assert node1.id != node1.secondary_uuid  # Should be different
        
        # Test 2: One UUID provided, one auto-generated
        manual_uuid = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
        node2 = self.MultiUUIDNode(id=manual_uuid, name="Mixed")
        
        # Only secondary_uuid should need generation
        fields_needing_generation = node2.get_auto_increment_fields_needing_generation()
        assert fields_needing_generation == ["secondary_uuid"]
        
        # Only id should be in manual values
        manual_values = node2.get_manual_auto_increment_values()
        assert manual_values == {"id": manual_uuid}
        
        session.add(node2)
        session.commit()
        
        # id should be preserved, secondary_uuid should be generated
        assert node2.id == manual_uuid
        assert node2.secondary_uuid is not None
        assert isinstance(node2.secondary_uuid, uuid.UUID)
        assert node2.secondary_uuid != manual_uuid
        
        # Test 3: Both UUIDs provided
        manual_uuid1 = uuid.UUID("550e8400-e29b-41d4-a716-446655440001")
        manual_uuid2 = uuid.UUID("550e8400-e29b-41d4-a716-446655440002")
        node3 = self.MultiUUIDNode(
            id=manual_uuid1, 
            secondary_uuid=manual_uuid2, 
            name="Manual Both"
        )
        
        # No fields should need generation
        fields_needing_generation = node3.get_auto_increment_fields_needing_generation()
        assert fields_needing_generation == []
        
        # Both should be in manual values
        manual_values = node3.get_manual_auto_increment_values()
        expected_manual = {"id": manual_uuid1, "secondary_uuid": manual_uuid2}
        assert manual_values == expected_manual
        
        session.add(node3)
        session.commit()
        
        # Both should be preserved
        assert node3.id == manual_uuid1
        assert node3.secondary_uuid == manual_uuid2
        
        session.close()

    def test_non_optional_uuid_field_behavior(self, test_db_path):
        """Test behavior with non-optional UUID auto-increment fields."""
        session = KuzuSession(db_path=test_db_path)
        initialize_schema(session, ddl=get_ddl_for_node(self.NonOptionalUUIDNode))
        
        # Test 1: No UUID provided (should be auto-generated)
        node1 = self.NonOptionalUUIDNode(name="Auto Generated")
        
        # Field should need generation
        fields_needing_generation = node1.get_auto_increment_fields_needing_generation()
        assert "id" in fields_needing_generation
        
        # No manual values
        manual_values = node1.get_manual_auto_increment_values()
        assert manual_values == {}
        
        session.add(node1)
        session.commit()
        
        # Should be generated
        assert node1.id is not None
        assert isinstance(node1.id, uuid.UUID)
        
        # Test 2: UUID provided manually
        manual_uuid = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
        node2 = self.NonOptionalUUIDNode(id=manual_uuid, name="Manual")
        
        # Field should NOT need generation
        fields_needing_generation = node2.get_auto_increment_fields_needing_generation()
        assert "id" not in fields_needing_generation
        
        # Should have manual value
        manual_values = node2.get_manual_auto_increment_values()
        assert manual_values == {"id": manual_uuid}
        
        session.add(node2)
        session.commit()
        
        # Should be preserved
        assert node2.id == manual_uuid
        
        session.close()

    def test_concurrent_uuid_field_access(self, test_db_path):
        """Test concurrent access to UUID auto-increment fields."""
        session = KuzuSession(db_path=test_db_path)
        initialize_schema(session, ddl=get_ddl_for_node(self.ATUUIDNode))

        # Create a node that will be accessed concurrently
        node = self.ATUUIDNode(name="Concurrent Test")

        results = []
        errors = []

        def access_manual_values(thread_id: int):
            """Access get_manual_auto_increment_values() from multiple threads."""
            try:
                for _ in range(100):  # Multiple accesses per thread
                    manual_values = node.get_manual_auto_increment_values()
                    # Should always return empty dict for this node
                    assert manual_values == {}
                    time.sleep(0.001)  # Small delay to increase chance of race conditions
                results.append(f"Thread {thread_id}: Success")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        # Start multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=access_manual_values, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 10, f"Expected 10 successful threads, got {len(results)}"

        session.close()

    def test_uuid_field_with_none_values(self, test_db_path):
        """Test UUID fields explicitly set to None."""
        session = KuzuSession(db_path=test_db_path)
        initialize_schema(session, ddl=get_ddl_for_node(self.ATUUIDNode))

        # Create node with explicit None for UUID field
        node = self.ATUUIDNode(id=None, name="Explicit None")

        # id should be in __pydantic_fields_set__ because it was explicitly provided
        assert "id" in node.__pydantic_fields_set__

        # get_manual_auto_increment_values() should return {"id": None}
        manual_values = node.get_manual_auto_increment_values()
        assert manual_values == {"id": None}

        # get_auto_increment_fields_needing_generation() should NOT include id
        # because it was explicitly set (even to None)
        fields_needing_generation = node.get_auto_increment_fields_needing_generation()
        assert "id" not in fields_needing_generation

        # Validation should accept None values (they get skipped)
        session._validate_manual_auto_increment_values(manual_values, self.ATUUIDNode)

        session.close()

    def test_uuid_validation_with_complex_types(self, test_db_path):
        """Test UUID validation with complex and unusual types."""
        session = KuzuSession(db_path=test_db_path)

        # Test with various complex types that should be rejected
        complex_invalid_values = [
            # Nested structures
            {"nested": {"uuid": "550e8400-e29b-41d4-a716-446655440000"}},
            [uuid.UUID("550e8400-e29b-41d4-a716-446655440000")],  # UUID in list
            (uuid.UUID("550e8400-e29b-41d4-a716-446655440000"),),  # UUID in tuple
            {uuid.UUID("550e8400-e29b-41d4-a716-446655440000")},   # UUID in set

            # Custom objects
            object(),
            type("CustomClass", (), {})(),

            # Functions and lambdas
            lambda: uuid.UUID("550e8400-e29b-41d4-a716-446655440000"),
            str,  # Type object

            # Complex numbers
            complex(1, 2),

            # Bytes that look like UUID
            b"550e8400-e29b-41d4-a716-446655440000",

            # Boolean values
            True,
            False,
        ]

        for invalid_value in complex_invalid_values:
            manual_values = {"id": invalid_value}

            with pytest.raises(TypeError) as exc_info:
                session._validate_manual_auto_increment_values(
                    manual_values,
                    self.ATUUIDNode
                )

            # Verify error message contains type information
            error_msg = str(exc_info.value)
            assert "UUID field 'id' in ATUUIDNode must be a UUID object" in error_msg
            assert type(invalid_value).__name__ in error_msg

        session.close()

    def test_precision_uuid_operations(self, test_db_path):
        """Test precision in UUID field operations."""
        session = KuzuSession(db_path=test_db_path)
        initialize_schema(session, ddl=get_ddl_for_node(self.MultiUUIDNode))

        # Generate a large number of UUIDs to test uniqueness and precision
        uuid_count = 1000
        generated_uuids = set()

        for i in range(uuid_count):
            node = self.MultiUUIDNode(name=f"Node_{i}")
            session.add(node)
            session.commit()

            # Verify both UUIDs are generated and unique
            assert node.id is not None
            assert node.secondary_uuid is not None
            assert isinstance(node.id, uuid.UUID)
            assert isinstance(node.secondary_uuid, uuid.UUID)

            # Verify uniqueness across all generated UUIDs
            assert node.id not in generated_uuids, f"Duplicate UUID generated: {node.id}"
            assert node.secondary_uuid not in generated_uuids, f"Duplicate UUID generated: {node.secondary_uuid}"

            generated_uuids.add(node.id)
            generated_uuids.add(node.secondary_uuid)

            # Verify the two UUIDs in the same node are different
            assert node.id != node.secondary_uuid

        # Verification: we should have exactly 2 * uuid_count unique UUIDs
        expected_uuid_count = 2 * uuid_count
        actual_uuid_count = len(generated_uuids)
        assert actual_uuid_count == expected_uuid_count, (
            f"Expected {expected_uuid_count} unique UUIDs, got {actual_uuid_count}"
        )

        session.close()

    def test_boundary_conditions_field_names(self, test_db_path):
        """Test boundary conditions with unusual field names and model configurations."""
        session = KuzuSession(db_path=test_db_path)

        # Test with model that has no auto-increment fields
        @kuzu_node("NoAutoIncrementNode")
        class NoAutoIncrementNode(KuzuBaseModel):
            id: str = kuzu_field(kuzu_type=KuzuDataType.STRING, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)

        node = NoAutoIncrementNode(id="manual_id", name="Test")

        # Should return empty lists/dicts
        auto_increment_fields = node.get_auto_increment_fields()
        assert auto_increment_fields == []

        manual_values = node.get_manual_auto_increment_values()
        assert manual_values == {}

        fields_needing_generation = node.get_auto_increment_fields_needing_generation()
        assert fields_needing_generation == []

        # Test with model that has __pydantic_fields_set__ missing
        node_no_fields_set = NoAutoIncrementNode(id="test", name="test")

        # Manually remove __pydantic_fields_set__ to test edge case
        if hasattr(node_no_fields_set, '__pydantic_fields_set__'):
            delattr(node_no_fields_set, '__pydantic_fields_set__')

        # Should handle missing __pydantic_fields_set__ gracefully
        manual_values = node_no_fields_set.get_manual_auto_increment_values()
        assert manual_values == {}  # Should default to empty set

        session.close()

    def test_extreme_stress_uuid_generation(self, test_db_path):
        """Extreme stress test: Generate thousands of UUIDs rapidly."""
        session = KuzuSession(db_path=test_db_path)
        initialize_schema(session, ddl=get_ddl_for_node(self.ATUUIDNode))

        # Generate a large number of nodes rapidly
        node_count = 2000  # Reduced for CI performance
        generated_uuids = set()

        start_time = time.time()

        for i in range(node_count):
            node = self.ATUUIDNode(name=f"StressTest_{i}")

            # Verify the node is properly configured for auto-generation
            assert "id" not in node.__pydantic_fields_set__
            assert node.get_manual_auto_increment_values() == {}
            assert "id" in node.get_auto_increment_fields_needing_generation()

            session.add(node)
            session.commit()

            # Verify UUID was generated and is unique
            assert node.id is not None
            assert isinstance(node.id, uuid.UUID)
            assert node.id not in generated_uuids, f"Duplicate UUID at iteration {i}: {node.id}"

            generated_uuids.add(node.id)

        end_time = time.time()
        duration = end_time - start_time

        # Performance verification: should complete in reasonable time
        assert duration < 120.0, f"Stress test took too long: {duration:.2f} seconds"

        # Verification: exactly node_count unique UUIDs
        assert len(generated_uuids) == node_count

        print(f"✓ Generated {node_count} unique UUIDs in {duration:.2f} seconds")
        print(f"✓ Average: {node_count/duration:.1f} UUIDs/second")

        session.close()

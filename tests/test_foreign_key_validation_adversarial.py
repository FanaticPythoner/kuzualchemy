# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
FOREIGN KEY VALIDATION TEST SUITE
==============================================

This test suite provides comprehensive testing for the automatic
foreign key validation system implemented in KuzuNodeBase. It tests:

1. Complex real-world Python code patterns
2. Edge cases that could cause failures
3. Boundary conditions and error scenarios
4. Nested structures and complex decorators
5. Performance stress cases with deeply nested code
6. Circular dependency handling
7. Cache invalidation and performance
8. Registry state management
"""

from __future__ import annotations

import pytest
import time
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from kuzualchemy.kuzu_orm import (
    KuzuNodeBase,
    kuzu_node,
    kuzu_field,
    KuzuDataType,
    ForeignKeyReference,
    clear_registry,
    _kuzu_registry,
)


class TestForeignKeyValidation:
    """Test suite for automatic foreign key validation."""
    
    def setup_method(self):
        """Clear registry before each test."""
        clear_registry()
    
    def teardown_method(self):
        """Clear registry after each test."""
        clear_registry()
    
    def test_automatic_validation_on_instantiation(self):
        """Test that foreign key validation happens automatically on node instantiation."""
        @kuzu_node("ValidTarget")
        class ValidTarget(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
        
        @kuzu_node("ValidReferrer")
        class ValidReferrer(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            target_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(
                    target_model=ValidTarget,
                    target_field="id"
                )
            )
        
        # @@ STEP: Finalize registry to enable validation
        _kuzu_registry.finalize_registry()
        
        # @@ STEP: This should succeed - valid foreign key reference
        valid_instance = ValidReferrer(id=1, target_id=100)
        assert valid_instance.id == 1
        assert valid_instance.target_id == 100
    
    def test_automatic_validation_with_invalid_foreign_key(self):
        """Test automatic validation catches invalid foreign key references."""
        @kuzu_node("InvalidTarget")
        class InvalidTarget(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
        
        @kuzu_node("InvalidReferrer")
        class InvalidReferrer(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            target_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(
                    target_model=InvalidTarget,
                    target_field="nonexistent_field"  # Invalid field
                )
            )
        
        # @@ STEP: Finalize registry to enable validation
        _kuzu_registry.finalize_registry()
        
        # @@ STEP: This should raise ValueError due to invalid foreign key
        with pytest.raises(ValueError) as exc_info:
            InvalidReferrer(id=1, target_id=100)
        
        # @@ STEP: Verify error message contains foreign key validation details
        error_msg = str(exc_info.value)
        assert "Foreign key validation failed" in error_msg
        assert "nonexistent_field" in error_msg
    
    def test_cache_performance_with_repeated_instantiation(self):
        """Test that caching provides performance benefits for repeated instantiation."""
        @kuzu_node("CacheTarget")
        class CacheTarget(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
        
        @kuzu_node("CacheReferrer")
        class CacheReferrer(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            target_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(
                    target_model=CacheTarget,
                    target_field="id"
                )
            )
        
        # @@ STEP: Finalize registry to enable validation
        _kuzu_registry.finalize_registry()
        
        # @@ STEP: Measure time for first instantiation (cache miss)
        start_time = time.time()
        first_instance = CacheReferrer(id=1, target_id=100)
        first_time = time.time() - start_time
        
        # @@ STEP: Measure time for subsequent instantiations (cache hits)
        times = []
        for i in range(10):
            start_time = time.time()
            CacheReferrer(id=i+2, target_id=100+i)
            times.append(time.time() - start_time)
        
        avg_cached_time = sum(times) / len(times)
        
        # @@ STEP: Cached instantiations should be faster (or at least not significantly slower)
        # || S.S: Allow some variance due to system load, but cached should generally be faster
        assert avg_cached_time <= first_time * 2.0, f"Cached time {avg_cached_time} should be <= {first_time * 2.0}"
        assert first_instance.id == 1
    
    def test_circular_dependency_handling(self):
        """Test that circular foreign key dependencies are handled gracefully."""
        @kuzu_node("CircularA")
        class CircularA(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            b_id: Optional[int] = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(
                    target_model="CircularB",  # Forward reference
                    target_field="id"
                ),
                default=None
            )
        
        @kuzu_node("CircularB")
        class CircularB(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            a_id: Optional[int] = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(
                    target_model=CircularA,  # Circular reference
                    target_field="id"
                ),
                default=None
            )
        
        # @@ STEP: Finalize registry to resolve circular dependencies
        success = _kuzu_registry.finalize_registry()
        assert success, "Registry should handle circular dependencies"
        
        # @@ STEP: Both classes should be instantiable
        a_instance = CircularA(id=1, b_id=None)
        b_instance = CircularB(id=2, a_id=1)
        
        assert a_instance.id == 1
        assert b_instance.id == 2
        assert b_instance.a_id == 1
    
    def test_concurrent_instantiation_thread_safety(self):
        """Test thread safety of foreign key validation during concurrent instantiation."""
        @kuzu_node("ThreadTarget")
        class ThreadTarget(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
        
        @kuzu_node("ThreadReferrer")
        class ThreadReferrer(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            target_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(
                    target_model=ThreadTarget,
                    target_field="id"
                )
            )
        
        # @@ STEP: Finalize registry to enable validation
        _kuzu_registry.finalize_registry()
        
        def create_instance(thread_id: int) -> ThreadReferrer:
            """Create an instance in a thread."""
            return ThreadReferrer(id=thread_id, target_id=thread_id * 10)
        
        # @@ STEP: Create instances concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_instance, i) for i in range(50)]
            instances = []
            
            for future in as_completed(futures):
                try:
                    instance = future.result(timeout=5.0)
                    instances.append(instance)
                except Exception as e:
                    pytest.fail(f"Concurrent instantiation failed: {e}")
        
        # @@ STEP: Verify all instances were created successfully
        assert len(instances) == 50
        ids = [instance.id for instance in instances]
        assert len(set(ids)) == 50  # All unique IDs
    
    def test_registry_state_change_invalidates_cache(self):
        """Test that registry state changes properly invalidate the validation cache."""
        @kuzu_node("StateTarget1")
        class StateTarget1(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
        
        @kuzu_node("StateReferrer1")
        class StateReferrer1(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            target_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(
                    target_model=StateTarget1,
                    target_field="id"
                )
            )
        
        # @@ STEP: Finalize registry and create instance (populates cache)
        _kuzu_registry.finalize_registry()
        first_instance = StateReferrer1(id=1, target_id=100)
        
        # @@ STEP: Register a new node (should invalidate cache)
        @kuzu_node("StateTarget2")
        class StateTarget2(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            value: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
        
        # @@ STEP: Create another instance (should work with fresh validation)
        second_instance = StateReferrer1(id=2, target_id=200)
        
        assert first_instance.id == 1
        assert second_instance.id == 2
    
    def test_complex_nested_foreign_key_validation(self):
        """Test validation with complex nested foreign key relationships."""
        @kuzu_node("Country")
        class Country(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
        
        @kuzu_node("State")
        class State(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
            country_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(target_model=Country, target_field="id")
            )
        
        @kuzu_node("City")
        class City(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
            state_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(target_model=State, target_field="id")
            )
            country_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(target_model=Country, target_field="id")
            )
        
        @kuzu_node("Address")
        class Address(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            street: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
            city_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(target_model=City, target_field="id")
            )
            state_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(target_model=State, target_field="id")
            )
            country_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(target_model=Country, target_field="id")
            )
        
        # @@ STEP: Finalize registry to enable validation
        _kuzu_registry.finalize_registry()
        
        # @@ STEP: Create instances with valid nested foreign keys
        address = Address(
            id=1,
            street="123 Main St",
            city_id=100,
            state_id=10,
            country_id=1
        )
        
        # @@ STEP: Verify all foreign key validations passed
        assert address.id == 1
        assert address.city_id == 100
        assert address.state_id == 10
        assert address.country_id == 1

    def test_validation_with_string_forward_references(self):
        """Test validation works correctly with string forward references."""
        @kuzu_node("ForwardReferrer")
        class ForwardReferrer(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            target_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(
                    target_model="ForwardTarget",  # String reference
                    target_field="id"
                )
            )

        @kuzu_node("ForwardTarget")
        class ForwardTarget(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)

        # @@ STEP: Finalize registry to resolve string references
        success = _kuzu_registry.finalize_registry()
        assert success, "Registry should resolve string references"

        # @@ STEP: Create instance with forward reference
        referrer = ForwardReferrer(id=1, target_id=100)
        assert referrer.id == 1
        assert referrer.target_id == 100

    def test_validation_during_registration_phase_skipped(self):
        """Test that validation is skipped during registration phase to avoid circular deps."""
        # @@ STEP: Clear registry to start fresh
        clear_registry()

        # @@ STEP: Create nodes during registration phase (validation should be skipped)
        @kuzu_node("RegistrationTarget")
        class RegistrationTarget(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)

        @kuzu_node("RegistrationReferrer")
        class RegistrationReferrer(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            target_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(
                    target_model=RegistrationTarget,
                    target_field="id"
                )
            )

        # @@ STEP: During registration phase, instantiation should work without validation
        # || S.S: This tests that we avoid circular dependencies during registration
        instance = RegistrationReferrer(id=1, target_id=100)
        assert instance.id == 1
        assert instance.target_id == 100

    def test_validation_error_handling_graceful_degradation(self):
        """Test graceful error handling when validation encounters unexpected errors."""
        @kuzu_node("ErrorTarget")
        class ErrorTarget(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)

        @kuzu_node("ErrorReferrer")
        class ErrorReferrer(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            target_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(
                    target_model=ErrorTarget,
                    target_field="id"
                )
            )

        # @@ STEP: Finalize registry
        _kuzu_registry.finalize_registry()

        # @@ STEP: Temporarily break the validation method to test error handling
        original_validate = ErrorReferrer.validate_foreign_keys

        def broken_validate():
            raise RuntimeError("Simulated validation error")

        ErrorReferrer.validate_foreign_keys = classmethod(lambda cls: broken_validate())

        try:
            # @@ STEP: This should handle the error gracefully and log a warning
            instance = ErrorReferrer(id=1, target_id=100)
            # @@ STEP: Instance should still be created despite validation error
            assert instance.id == 1
            assert instance.target_id == 100
        finally:
            # @@ STEP: Restore original validation method
            ErrorReferrer.validate_foreign_keys = original_validate

    def test_cache_size_management_with_many_nodes(self):
        """Test that cache size management works correctly with many different nodes."""
        # @@ STEP: Create many node classes to test cache eviction
        node_classes = []

        for i in range(20):  # Create more than cache size to test eviction
            @kuzu_node(f"CacheNode{i}")
            class CacheNode(KuzuNodeBase):
                id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
                name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)

            node_classes.append(CacheNode)

        # @@ STEP: Create referrer nodes for each target
        referrer_classes = []
        for i, target_class in enumerate(node_classes):
            @kuzu_node(f"CacheReferrer{i}")
            class CacheReferrer(KuzuNodeBase):
                id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
                target_id: int = kuzu_field(
                    kuzu_type=KuzuDataType.INT64,
                    foreign_key=ForeignKeyReference(
                        target_model=target_class,
                        target_field="id"
                    )
                )

            referrer_classes.append(CacheReferrer)

        # @@ STEP: Finalize registry
        _kuzu_registry.finalize_registry()

        # @@ STEP: Create instances to populate and test cache
        instances = []
        for i, referrer_class in enumerate(referrer_classes):
            instance = referrer_class(id=i, target_id=i * 10)
            instances.append(instance)

        # @@ STEP: Verify all instances were created successfully
        assert len(instances) == 20
        for i, instance in enumerate(instances):
            assert instance.id == i
            assert instance.target_id == i * 10

    def test_validation_with_optional_foreign_keys(self):
        """Test validation works correctly with optional foreign key fields."""
        @kuzu_node("OptionalTarget")
        class OptionalTarget(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)

        @kuzu_node("OptionalReferrer")
        class OptionalReferrer(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            optional_target_id: Optional[int] = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(
                    target_model=OptionalTarget,
                    target_field="id"
                ),
                default=None
            )

        # @@ STEP: Finalize registry
        _kuzu_registry.finalize_registry()

        # @@ STEP: Create instance with None foreign key (should work)
        instance_none = OptionalReferrer(id=1, optional_target_id=None)
        assert instance_none.id == 1
        assert instance_none.optional_target_id is None

        # @@ STEP: Create instance with valid foreign key (should work)
        instance_valid = OptionalReferrer(id=2, optional_target_id=100)
        assert instance_valid.id == 2
        assert instance_valid.optional_target_id == 100

    def test_validation_performance_stress_test(self):
        """Stress test validation performance with rapid instantiation."""
        @kuzu_node("StressTarget")
        class StressTarget(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)

        @kuzu_node("StressReferrer")
        class StressReferrer(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            target_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(
                    target_model=StressTarget,
                    target_field="id"
                )
            )

        # @@ STEP: Finalize registry
        _kuzu_registry.finalize_registry()

        # @@ STEP: Create many instances rapidly to stress test caching
        start_time = time.time()
        instances = []

        for i in range(1000):  # Create 1000 instances
            instance = StressReferrer(id=i, target_id=i * 10)
            instances.append(instance)

        total_time = time.time() - start_time

        # @@ STEP: Verify all instances were created
        assert len(instances) == 1000

        # @@ STEP: Performance should be reasonable (less than 5 seconds for 1000 instances)
        assert total_time < 5.0, f"Stress test took {total_time} seconds, should be < 5.0"

        # @@ STEP: Verify correctness of created instances
        for i, instance in enumerate(instances):
            assert instance.id == i
            assert instance.target_id == i * 10

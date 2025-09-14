# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
Tests for KuzuNodeBase implementation.

This test suite is designed to break the KuzuNodeBase implementation by testing:
- Complex real-world Python code patterns
- Edge cases that could cause failures
- Boundary conditions and error scenarios
- Nested structures, multiple inheritance, complex decorators
- Unusual but valid Python syntax
- Performance stress cases with deeply nested code

All tests use equality-only assertions with exact expected values and
step-by-step derivation comments showing justification.
"""

from __future__ import annotations

import pytest
import uuid
import datetime
from decimal import Decimal
from typing import Optional
from pydantic import ValidationError

from kuzualchemy import (
    KuzuNodeBase,
    KuzuBaseModel,
    KuzuRelationshipBase,
    kuzu_node,
    kuzu_relationship,
    kuzu_field,
    KuzuDataType,
    clear_registry,
)
from kuzualchemy.constants import NodeBaseConstants, ModelMetadataConstants


class TestKuzuNodeBaseBasicFunctionality:
    """Test basic KuzuNodeBase functionality with exact assertions."""
    
    def setup_method(self):
        """Set up clean registry for each test."""
        clear_registry()
    
    def test_kuzu_node_base_inheritance(self):
        """Test that KuzuNodeBase properly inherits from KuzuBaseModel."""
        # Expected: KuzuNodeBase should inherit from KuzuBaseModel
        # Derivation: KuzuNodeBase.__bases__ should contain KuzuBaseModel
        assert KuzuNodeBase.__bases__ == (KuzuBaseModel,)
        
        # Expected: KuzuNodeBase should be a subclass of KuzuBaseModel
        # Derivation: issubclass(KuzuNodeBase, KuzuBaseModel) should be True
        assert issubclass(KuzuNodeBase, KuzuBaseModel)
    
    def test_kuzu_node_base_marker_attribute(self):
        """Test that KuzuNodeBase has the correct marker attribute."""
        # Expected: KuzuNodeBase should have __is_kuzu_node_base__ = True
        # Derivation: getattr(KuzuNodeBase, '__is_kuzu_node_base__', False) should be True
        assert getattr(KuzuNodeBase, '__is_kuzu_node_base__', False)
    
    def test_is_node_base_class_method(self):
        """Test the is_node_base class method."""
        # Expected: KuzuNodeBase.is_node_base() should return True
        # Derivation: Method should check for __is_kuzu_node_base__ attribute
        assert KuzuNodeBase.is_node_base()

        # Expected: KuzuBaseModel no longer has is_node_base method
        # Derivation: Clean API - only KuzuNodeBase has this method
        assert not hasattr(KuzuBaseModel, 'is_node_base')
    
    def test_node_base_with_decorator(self):
        """Test KuzuNodeBase with @kuzu_node decorator."""
        @kuzu_node("TestNode")
        class TestNode(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
        
        # Expected: TestNode should have node name attribute
        # Derivation: @kuzu_node decorator should set __kuzu_node_name__
        assert hasattr(TestNode, ModelMetadataConstants.KUZU_NODE_NAME) == True
        assert getattr(TestNode, ModelMetadataConstants.KUZU_NODE_NAME) == "TestNode"
        
        # Expected: TestNode.get_node_name() should return "TestNode"
        # Derivation: Method should retrieve __kuzu_node_name__ attribute
        assert TestNode.get_node_name() == "TestNode"
        
        # Expected: TestNode should still be a node base
        # Derivation: Inheritance should be preserved
        assert TestNode.is_node_base() == True


class TestKuzuNodeBaseEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def setup_method(self):
        """Set up clean registry for each test."""
        clear_registry()
    
    def test_node_without_decorator_validation(self):
        """Test validation of nodes without @kuzu_node decorator."""
        class UnDecoratedNode(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        # Expected: validate_node_decoration should raise ValueError
        # Derivation: Node without @kuzu_node should fail validation
        with pytest.raises(ValueError) as exc_info:
            UnDecoratedNode.validate_node_decoration()
        
        # Expected: Error message should contain class name
        # Derivation: NodeBaseConstants.NODE_MISSING_DECORATOR format string
        expected_message = NodeBaseConstants.NODE_MISSING_DECORATOR.format("UnDecoratedNode")
        assert str(exc_info.value) == expected_message
    
    def test_get_node_name_without_decorator(self):
        """Test get_node_name on undecorated node."""
        class UnDecoratedNode(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        # Expected: get_node_name should raise ValueError
        # Derivation: Method calls validate_node_decoration which should fail
        with pytest.raises(ValueError) as exc_info:
            UnDecoratedNode.get_node_name()
        
        expected_message = NodeBaseConstants.NODE_MISSING_DECORATOR.format("UnDecoratedNode")
        assert str(exc_info.value) == expected_message
    
    def test_node_without_primary_key_validation(self):
        """Test validation of nodes without primary key fields."""
        @kuzu_node("NoPKNode")
        class NoPKNode(KuzuNodeBase):
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)

        # Expected: Node without primary key should fail at instantiation time
        # Derivation: Pydantic model validator should catch this during creation
        with pytest.raises(ValidationError) as exc_info:
            NoPKNode(name="test")

        # Expected: Error should indicate missing primary key
        # Derivation: Pydantic validation error should contain relevant message
        error_str = str(exc_info.value)
        assert "primary key" in error_str.lower()


class TestKuzuNodeBaseComplexScenarios:
    """Test complex real-world scenarios and unusual patterns."""
    
    def setup_method(self):
        """Set up clean registry for each test."""
        clear_registry()
    
    def test_multiple_inheritance_with_node_base(self):
        """Test multiple inheritance patterns with KuzuNodeBase."""
        class Mixin:
            def mixin_method(self) -> str:
                return "mixin"
        
        @kuzu_node("MultiInheritNode")
        class MultiInheritNode(KuzuNodeBase, Mixin):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
        
        # Expected: Node should work with multiple inheritance
        # Derivation: Python MRO should resolve correctly
        instance = MultiInheritNode(id=1, name="test")
        
        # Expected: Both base classes should be accessible
        # Derivation: MRO should include both KuzuNodeBase and Mixin
        assert isinstance(instance, KuzuNodeBase) == True
        assert isinstance(instance, Mixin) == True
        assert instance.mixin_method() == "mixin"
        assert instance.get_node_name() == "MultiInheritNode"

    def test_node_base_with_complex_primary_key_types(self):
        """Test KuzuNodeBase with various complex primary key types."""
        @kuzu_node("UUIDNode")
        class UUIDNode(KuzuNodeBase):
            id: uuid.UUID = kuzu_field(kuzu_type=KuzuDataType.UUID, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)

        @kuzu_node("DecimalNode")
        class DecimalNode(KuzuNodeBase):
            id: Decimal = kuzu_field(kuzu_type=KuzuDataType.DECIMAL, primary_key=True)
            value: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE)

        @kuzu_node("DateTimeNode")
        class DateTimeNode(KuzuNodeBase):
            id: datetime.datetime = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)

        # Expected: All complex PK types should work
        # Derivation: KuzuNodeBase should handle any valid Kuzu primary key type
        uuid_id = uuid.uuid4()
        uuid_node = UUIDNode(id=uuid_id, name="uuid_test")
        assert uuid_node.get_node_name() == "UUIDNode"

        decimal_id = Decimal("123.45")
        decimal_node = DecimalNode(id=decimal_id, value=67.89)
        assert decimal_node.get_node_name() == "DecimalNode"

        dt_id = datetime.datetime.now()
        dt_node = DateTimeNode(id=dt_id, name="datetime_test")
        assert dt_node.get_node_name() == "DateTimeNode"

    def test_node_base_in_relationships_with_instances(self):
        """Test KuzuNodeBase instances in relationships."""
        @kuzu_node("PersonNode")
        class PersonNode(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)

        @kuzu_relationship("KNOWS", pairs=[(PersonNode, PersonNode)])
        class KnowsRel(KuzuRelationshipBase):
            since: int = kuzu_field(kuzu_type=KuzuDataType.INT32, default=2024)

        # Expected: Relationship should accept KuzuNodeBase instances
        # Derivation: NodeReference type should allow KuzuNodeBase instances
        person1 = PersonNode(id=1, name="Alice")
        person2 = PersonNode(id=2, name="Bob")

        rel = KnowsRel(from_node=person1, to_node=person2, since=2020)

        # Expected: Relationship should store node instances correctly
        # Derivation: from_node and to_node should be the exact instances
        assert rel.from_node is person1
        assert rel.to_node is person2
        assert rel.since == 2020

    def test_node_base_in_relationships_with_raw_values(self):
        """Test KuzuNodeBase relationships with raw primary key values."""
        @kuzu_node("ProductNode")
        class ProductNode(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)

        @kuzu_relationship("RELATED", pairs=[(ProductNode, ProductNode)])
        class RelatedRel(KuzuRelationshipBase):
            strength: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, default=1.0)

        # Expected: Relationship should accept raw primary key values
        # Derivation: NodeReference type should allow primitive PK types
        rel = RelatedRel(from_node=100, to_node=200, strength=0.75)

        # Expected: Raw values should be stored as-is
        # Derivation: No conversion should occur for raw values
        assert rel.from_node == 100
        assert rel.to_node == 200
        assert rel.strength == 0.75

    def test_node_base_with_abstract_inheritance(self):
        """Test KuzuNodeBase with abstract node inheritance."""
        @kuzu_node("AbstractBase", abstract=True)
        class AbstractBaseNode(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            created_at: datetime.datetime = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP)

        @kuzu_node("ConcreteNode")
        class ConcreteNode(AbstractBaseNode):
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)

        # Expected: Concrete node should inherit from abstract node base
        # Derivation: Python inheritance should work normally
        concrete = ConcreteNode(id=1, created_at=datetime.datetime.now(), name="test")

        # Expected: Concrete node should be a node base
        # Derivation: Inheritance should preserve node base properties
        assert concrete.is_node_base() == True
        assert concrete.get_node_name() == "ConcreteNode"

        # Expected: Abstract base should still be a node base
        # Derivation: Abstract nodes are still node bases
        assert AbstractBaseNode.is_node_base() == True


class TestKuzuNodeBasePerformanceStress:
    """Test performance stress cases with deeply nested code."""

    def setup_method(self):
        """Set up clean registry for each test."""
        clear_registry()

    def test_large_number_of_node_classes(self):
        """Test creating a large number of node classes."""
        node_classes = []

        # Create 100 node classes
        for i in range(100):
            @kuzu_node(f"StressNode{i}")
            class StressNode(KuzuNodeBase):
                id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
                value: int = kuzu_field(kuzu_type=KuzuDataType.INT32, default=i)

            # Rename class to avoid conflicts
            StressNode.__name__ = f"StressNode{i}"
            StressNode.__qualname__ = f"StressNode{i}"
            node_classes.append(StressNode)

        # Expected: All node classes should be valid node bases
        # Derivation: Each class should inherit KuzuNodeBase properties
        for i, node_cls in enumerate(node_classes):
            assert node_cls.is_node_base() == True
            assert node_cls.get_node_name() == f"StressNode{i}"

            # Test instance creation
            instance = node_cls(id=i, value=i * 2)
            assert instance.id == i
            assert instance.value == i * 2


class TestKuzuNodeBaseErrorConditions:
    """Test error conditions and boundary cases that should fail gracefully."""

    def setup_method(self):
        """Set up clean registry for each test."""
        clear_registry()

    def test_node_base_with_invalid_primary_key_values(self):
        """Test node validation with invalid primary key values."""
        @kuzu_node("TestNode")
        class TestNode(KuzuNodeBase):
            id: Optional[int] = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)

        # Expected: Node with None primary key should fail at instantiation time
        # Derivation: Pydantic model validator should catch this during creation
        with pytest.raises(ValidationError) as exc_info:
            TestNode(id=None, name="test")

        # Expected: Error should indicate no primary key values set
        # Derivation: Pydantic validation error should contain relevant message
        error_str = str(exc_info.value)
        assert "primary key values set" in error_str

    def test_kuzu_node_base_only_api(self):
        """Test that only KuzuNodeBase nodes work in relationships."""
        @kuzu_node("NewStyleNode")
        class NewStyleNode(KuzuNodeBase):  # Using new base class
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)

        @kuzu_relationship("NEW_REL", pairs=[(NewStyleNode, NewStyleNode)])
        class NewRel(KuzuRelationshipBase):
            weight: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, default=1.0)

        # Expected: KuzuNodeBase nodes should work in relationships
        # Derivation: NodeReference type includes KuzuNodeBase
        new_node1 = NewStyleNode(id=1, name="new1")
        new_node2 = NewStyleNode(id=2, name="new2")

        rel = NewRel(from_node=new_node1, to_node=new_node2, weight=0.5)

        # Expected: Relationship should work with KuzuNodeBase nodes
        # Derivation: Clean API
        assert rel.from_node is new_node1
        assert rel.to_node is new_node2
        assert rel.weight == 0.5

        # Expected: KuzuNodeBase node should be a node base
        # Derivation: KuzuNodeBase has __is_kuzu_node_base__ = True
        assert new_node1.is_node_base()

    def test_kuzu_node_base_with_raw_primary_keys(self):
        """Test KuzuNodeBase relationships with raw primary key values."""
        @kuzu_node("CleanNode")
        class CleanNode(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)

        @kuzu_relationship("CLEAN_REL", pairs=[(CleanNode, CleanNode)])
        class CleanRel(KuzuRelationshipBase):
            type: str = kuzu_field(kuzu_type=KuzuDataType.STRING, default="clean")

        # Expected: Relationships should work with KuzuNodeBase instances
        # Derivation: NodeReference includes KuzuNodeBase
        clean_node1 = CleanNode(id=1, name="clean1")
        clean_node2 = CleanNode(id=2, name="clean2")

        rel1 = CleanRel(from_node=clean_node1, to_node=clean_node2, type="node_to_node")

        # Expected: Relationships should work with raw primary key values
        # Derivation: NodeReference includes primary key types
        rel2 = CleanRel(from_node=100, to_node=200, type="raw_to_raw")

        # Expected: Mixed node instances and raw values should work
        # Derivation: NodeReference Union includes both types
        rel3 = CleanRel(from_node=clean_node1, to_node=300, type="node_to_raw")

        # Verify all relationships work correctly
        assert rel1.from_node is clean_node1
        assert rel1.to_node is clean_node2
        assert rel1.type == "node_to_node"

        assert rel2.from_node == 100
        assert rel2.to_node == 200
        assert rel2.type == "raw_to_raw"

        assert rel3.from_node is clean_node1
        assert rel3.to_node == 300
        assert rel3.type == "node_to_raw"

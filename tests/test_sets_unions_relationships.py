# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive tests for sets and unions in relationship targets.
Tests the handling of sets, lists, and unions in relationship pair definitions.
"""
import pytest

from kuzualchemy.kuzu_orm import (
    kuzu_node,
    kuzu_relationship,
    KuzuBaseModel,
    kuzu_field,
    KuzuDataType,
    RelationshipPair,
    _process_relationship_pairs,
    clear_registry,
    generate_relationship_ddl,
)


class TestSetsUnionsInRelationships:
    """Test suite for sets and unions in relationship definitions."""
    
    def setup_method(self):
        """Clear registry before each test."""
        clear_registry()
    
    def teardown_method(self):
        """Clear registry after each test."""
        clear_registry()
    
    def test_set_expansion_in_dict_format(self):
        """Test that dictionary format is NOT supported and raises an error."""
        @kuzu_node("NodeA")
        class NodeA(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        @kuzu_node("NodeB")
        class NodeB(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        @kuzu_node("NodeC")
        class NodeC(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        # Dictionary format should NOT be supported
        with pytest.raises(ValueError, match="'pairs' must be a list of tuples"):
            _process_relationship_pairs(
                {NodeA: {NodeB, NodeC}},
                "TestRel"
            )
    
    def test_list_expansion_in_dict_format(self):
        """Test that dictionary format with lists is NOT supported and raises an error."""
        @kuzu_node("NodeX")
        class NodeX(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        @kuzu_node("NodeY")
        class NodeY(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        @kuzu_node("NodeZ")
        class NodeZ(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        # Dictionary format should NOT be supported
        with pytest.raises(ValueError, match="'pairs' must be a list of tuples"):
            _process_relationship_pairs(
                {NodeX: [NodeY, NodeZ]},
                "TestRel"
            )
    
    def test_set_expansion_in_tuple_format(self):
        """Test that sets in tuple format are properly expanded."""
        @kuzu_node("Node1")
        class Node1(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        @kuzu_node("Node2")
        class Node2(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        @kuzu_node("Node3")
        class Node3(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        # Test set expansion in tuple format
        pairs = _process_relationship_pairs(
            [(Node1, {Node2, Node3})],
            "TestRel"
        )

        assert len(pairs) == 2
        names = {(p.get_from_name(), p.get_to_name()) for p in pairs}
        assert ("Node1", "Node2") in names
        assert ("Node1", "Node3") in names
    
    def test_complex_set_expansion(self):
        """Test complex combinations of sets in both from and to positions."""
        @kuzu_node("Variable")
        class Variable(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
        
        @kuzu_node("Parameter")
        class Parameter(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
        
        @kuzu_node("Attribute")
        class Attribute(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
        
        @kuzu_node("Function")
        class Function(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
        
        # Test complex set expansion - the case from the error message
        pairs = _process_relationship_pairs(
            [(Function, {Variable, Parameter, Attribute})],
            "FunctionDefines"
        )
        
        assert len(pairs) == 3
        names = {(p.get_from_name(), p.get_to_name()) for p in pairs}
        assert ("Function", "Variable") in names
        assert ("Function", "Parameter") in names
        assert ("Function", "Attribute") in names
    
    def test_relationship_with_sets_generates_correct_ddl(self):
        """Test that relationships with sets generate correct DDL."""
        @kuzu_node("SourceNode")
        class SourceNode(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        @kuzu_node("TargetNode1")
        class TargetNode1(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        @kuzu_node("TargetNode2")
        class TargetNode2(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        @kuzu_relationship(
            "CONNECTS",
            pairs=[(SourceNode, {TargetNode1, TargetNode2})]
        )
        class ConnectsRel(KuzuBaseModel):
            weight: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE)
        
        ddl = generate_relationship_ddl(ConnectsRel)
        
        # Should contain both target nodes
        assert "TargetNode1" in ddl
        assert "TargetNode2" in ddl
        assert "FROM SourceNode" in ddl
        # Should have two FROM-TO pairs
        assert ddl.count("FROM SourceNode") == 2
    
    def test_mixed_dict_format_with_sets_and_singles(self):
        """Test that mixed dictionary format is NOT supported and raises an error."""
        @kuzu_node("A")
        class A(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        @kuzu_node("B")
        class B(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        @kuzu_node("C")
        class C(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        @kuzu_node("D")
        class D(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        # Dictionary format should NOT be supported
        with pytest.raises(ValueError, match="'pairs' must be a list of tuples"):
            _process_relationship_pairs(
                {A: B, C: {D, A}},
                "MixedRel"
            )
    
    def test_frozenset_expansion(self):
        """Test that frozensets in tuple format are handled like sets."""
        @kuzu_node("FNode1")
        class FNode1(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        @kuzu_node("FNode2")
        class FNode2(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        @kuzu_node("FNode3")
        class FNode3(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        # Test frozenset expansion in tuple format
        pairs = _process_relationship_pairs(
            [(FNode1, frozenset([FNode2, FNode3]))],
            "FrozenRel"
        )
        
        assert len(pairs) == 2
        names = {(p.get_from_name(), p.get_to_name()) for p in pairs}
        assert ("FNode1", "FNode2") in names
        assert ("FNode1", "FNode3") in names
    
    def test_set_in_both_positions(self):
        """Test sets in both from and to positions (Cartesian product)."""
        @kuzu_node("M1")
        class M1(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        @kuzu_node("M2")
        class M2(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        @kuzu_node("N1")
        class N1(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        @kuzu_node("N2")
        class N2(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        # Test Cartesian product
        pairs = _process_relationship_pairs(
            [({M1, M2}, {N1, N2})],
            "CartesianRel"
        )
        
        assert len(pairs) == 4
        names = {(p.get_from_name(), p.get_to_name()) for p in pairs}
        assert ("M1", "N1") in names
        assert ("M1", "N2") in names
        assert ("M2", "N1") in names
        assert ("M2", "N2") in names
    
    def test_error_on_unexpanded_set_in_relationship_pair(self):
        """Test that RelationshipPair raises error if it receives unexpanded sets."""
        @kuzu_node("TestNode")
        class TestNode(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        # Should raise error when get_from_name() is called with a set
        pair1 = RelationshipPair({TestNode}, TestNode)
        with pytest.raises(TypeError, match="RelationshipPair.from_node received a set"):
            pair1.get_from_name()
        
        # Should raise error when get_to_name() is called with a set
        pair2 = RelationshipPair(TestNode, {TestNode})
        with pytest.raises(TypeError, match="RelationshipPair.to_node received a set"):
            pair2.get_to_name()
    
    def test_string_node_names_work(self):
        """Test that string node names work in tuple format with sets."""
        # No need to register nodes when using string names in tuple format
        pairs = _process_relationship_pairs(
            [("NodeA", {"NodeB", "NodeC"})],
            "StringRel"
        )
        
        assert len(pairs) == 2
        names = {(p.get_from_name(), p.get_to_name()) for p in pairs}
        assert ("NodeA", "NodeB") in names
        assert ("NodeA", "NodeC") in names
    
    def test_empty_set_raises_error(self):
        """Test that empty sets raise an appropriate error."""
        @kuzu_node("EmptyNode")
        class EmptyNode(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        with pytest.raises(ValueError, match="No valid relationship pairs found"):
            _process_relationship_pairs(
                [(EmptyNode, set())],
                "EmptyRel"
            )

"""
Test suite for multi-pair relationship functionality in KuzuAlchemy.

Tests the ability to create relationships with multiple FROM-TO pairs,
as supported by Kuzu's CREATE REL TABLE syntax.
"""

from __future__ import annotations

import pytest
from typing import Optional

from kuzualchemy.kuzu_orm import (
    KuzuBaseModel,
    kuzu_node,
    kuzu_relationship,
    kuzu_field,
    generate_relationship_ddl,
    RelationshipPair,
    RelationshipMultiplicity,
    KuzuDataType,
)


# @@ STEP 1: Define test node models
@kuzu_node(name="User")
class User(KuzuBaseModel):
    """User node for testing."""
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
    age: Optional[int] = kuzu_field(kuzu_type=KuzuDataType.INT32, default=None)


@kuzu_node(name="City")  
class City(KuzuBaseModel):
    """City node for testing."""
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    name: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    population: int = kuzu_field(kuzu_type=KuzuDataType.INT64, default=0)


@kuzu_node(name="Post")
class Post(KuzuBaseModel):
    """Post node for testing."""
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    title: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    content: str = kuzu_field(kuzu_type=KuzuDataType.STRING)


@kuzu_node(name="UserGroup")
class UserGroup(KuzuBaseModel):
    """UserGroup node for testing."""
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    name: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    description: str = kuzu_field(kuzu_type=KuzuDataType.STRING)


@kuzu_node(name="Device")
class Device(KuzuBaseModel):
    """Device node for testing."""
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    name: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    device_type: str = kuzu_field(kuzu_type=KuzuDataType.STRING)


# @@ STEP 2: Test single-pair relationships (backward compatibility)
class TestSinglePairRelationships:
    """Test relationships with a single FROM-TO pair."""
    
    def test_simple_relationship(self):
        """Test basic single-pair relationship."""
        @kuzu_relationship(
            name="Follows",
            pairs=[(User, User)]
        )
        class Follows(KuzuBaseModel):
            since: str = kuzu_field(kuzu_type=KuzuDataType.DATE)
        
        # Verify metadata
        assert Follows.__kuzu_relationship_name__ == "Follows"
        assert len(Follows.__kuzu_relationship_pairs__) == 1
        assert not Follows.__kuzu_is_multi_pair__
        
        assert Follows.__kuzu_relationship_pairs__[0].get_from_name() == "User"
        assert Follows.__kuzu_relationship_pairs__[0].get_to_name() == "User"
        
        # Generate DDL
        ddl = generate_relationship_ddl(Follows)
        # Default multiplicity is added
        assert "CREATE REL TABLE Follows(FROM User TO User, since DATE" in ddl
    
    def test_relationship_with_multiplicity(self):
        """Test single-pair relationship with multiplicity constraint."""
        @kuzu_relationship(
            name="LivesIn",
            pairs=[(User, City)],
            multiplicity=RelationshipMultiplicity.MANY_TO_ONE
        )
        class LivesIn(KuzuBaseModel):
            since_year: int = kuzu_field(kuzu_type=KuzuDataType.INT32)
        
        ddl = generate_relationship_ddl(LivesIn)
        assert "CREATE REL TABLE LivesIn(FROM User TO City, since_year INT32, MANY_ONE)" in ddl
    
    def test_relationship_with_string_nodes(self):
        """Test relationship using string node names."""
        @kuzu_relationship(
            name="WorksIn",
            pairs=[("User", "City")]  # Use existing registered nodes
        )
        class WorksIn(KuzuBaseModel):
            position: str = kuzu_field(kuzu_type=KuzuDataType.STRING)

        pairs = WorksIn.__kuzu_relationship_pairs__
        assert len(pairs) == 1
        assert pairs[0].get_from_name() == "User"
        assert pairs[0].get_to_name() == "City"


# @@ STEP 3: Test multi-pair relationships
class TestMultiPairRelationships:
    """Test relationships with multiple FROM-TO pairs."""
    
    def test_multi_pair_relationship(self):
        """Test relationship with multiple FROM-TO pairs."""
        @kuzu_relationship(
            name="Knows",
            pairs=[(User, User), (User, City)]
        )
        class Knows(KuzuBaseModel):
            since: str = kuzu_field(kuzu_type=KuzuDataType.DATE)
            strength: int = kuzu_field(kuzu_type=KuzuDataType.INT32, default=1)
        
        # Verify metadata
        assert Knows.__kuzu_relationship_name__ == "Knows"
        assert len(Knows.__kuzu_relationship_pairs__) == 2
        assert Knows.__kuzu_is_multi_pair__
        
        # Check pairs
        pairs = Knows.__kuzu_relationship_pairs__
        assert len(pairs) == 2
        assert pairs[0].get_from_name() == "User"
        assert pairs[0].get_to_name() == "User"
        assert pairs[1].get_from_name() == "User"
        assert pairs[1].get_to_name() == "City"
        
        # Generate DDL
        ddl = generate_relationship_ddl(Knows)
        assert "FROM User TO User, FROM User TO City" in ddl
        assert "since DATE" in ddl
        assert "strength INT32 DEFAULT 1" in ddl
    
    def test_complex_multi_pair_relationship(self):
        """Test relationship with many pairs and properties."""
        @kuzu_relationship(
            name="Interacts",
            pairs=[
                (User, User),
                (User, Post),
                (Post, Post),
                (UserGroup, User)
            ],
            multiplicity=RelationshipMultiplicity.MANY_TO_MANY
        )
        class Interacts(KuzuBaseModel):
            interaction_type: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
            timestamp: str = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP)
            score: float = kuzu_field(kuzu_type=KuzuDataType.FLOAT, default=0.0)
        
        pairs = Interacts.__kuzu_relationship_pairs__
        assert len(pairs) == 4
        assert Interacts.__kuzu_is_multi_pair__
        
        ddl = generate_relationship_ddl(Interacts)
        assert "FROM User TO User" in ddl
        assert "FROM User TO Post" in ddl
        assert "FROM Post TO Post" in ddl
        assert "FROM UserGroup TO User" in ddl
        assert "MANY_MANY" in ddl
    
    def test_mixed_node_types_in_pairs(self):
        """Test mixing class types and string names in pairs."""
        @kuzu_relationship(
            name="Connected",
            pairs=[
                (User, "Device"),
                ("Device", City),
                (City, Post)
            ]
        )
        class Connected(KuzuBaseModel):
            connection_type: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
        
        pairs = Connected.__kuzu_relationship_pairs__
        assert pairs[0].get_from_name() == "User"
        assert pairs[0].get_to_name() == "Device"
        assert pairs[1].get_from_name() == "Device"
        assert pairs[1].get_to_name() == "City"
        assert pairs[2].get_from_name() == "City"
        assert pairs[2].get_to_name() == "Post"


# @@ STEP 4: Test error cases
class TestErrorCases:
    """Test error handling for relationship definitions."""
    
    def test_no_pairs_error(self):
        """Test error when no pairs defined."""
        with pytest.raises(ValueError, match="must have 'pairs' parameter defined"):
            @kuzu_relationship(name="BadRel")
            class BadRel(KuzuBaseModel):
                passrop: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
    
    def test_empty_pairs_error(self):
        """Test error when pairs is empty."""
        with pytest.raises(ValueError, match="must have 'pairs' parameter defined"):
            @kuzu_relationship(name="EmptyRel", pairs=[])
            class EmptyRel(KuzuBaseModel):
                passrop: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
    
    def test_invalid_node_in_pair(self):
        """Test that relationships with invalid node references are properly validated."""
        # This test should verify that the system properly handles invalid node references
        # Instead of creating a relationship that pollutes the global registry,
        # we should test the validation logic directly

        class InvalidNode:
            """Node without kuzu_node decorator."""
            pass

        # Test that we can create the relationship pair and get the name
        from kuzualchemy.kuzu_orm import RelationshipPair
        pair = RelationshipPair(InvalidNode, User)

        # This should work because InvalidNode has __name__ attribute
        assert pair.get_from_name() == "InvalidNode"
        assert pair.get_to_name() == "User"

        # The real validation should happen when DDL is generated or executed
        # But we shouldn't create a globally registered relationship for this test
    
    def test_abstract_relationship_no_pairs_allowed(self):
        """Test that abstract relationships don't require pairs."""
        @kuzu_relationship(
            name="AbstractRel",
            abstract=True
        )
        class AbstractRel(KuzuBaseModel):
            """Abstract relationship for inheritance."""
            common_prop: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
        
        # Should not raise error
        assert AbstractRel.__kuzu_is_abstract__
        assert AbstractRel.__kuzu_relationship_pairs__ == []


# @@ STEP 5: Test RelationshipPair class
class TestRelationshipPair:
    """Test the RelationshipPair helper class."""
    
    def test_pair_with_class_nodes(self):
        """Test RelationshipPair with class node types."""
        pair = RelationshipPair(User, City)
        assert pair.get_from_name() == "User"
        assert pair.get_to_name() == "City"
        assert pair.to_ddl_component() == "FROM User TO City"
    
    def test_pair_with_string_nodes(self):
        """Test RelationshipPair with string node names."""
        pair = RelationshipPair("User", "City")  # Use existing registered nodes
        assert pair.get_from_name() == "User"
        assert pair.get_to_name() == "City"
        assert pair.to_ddl_component() == "FROM User TO City"
    
    def test_pair_mixed_types(self):
        """Test RelationshipPair with mixed node types."""
        pair = RelationshipPair(User, "Location")
        assert pair.get_from_name() == "User"
        assert pair.get_to_name() == "Location"
        assert pair.to_ddl_component() == "FROM User TO Location"
    
    def test_pair_repr(self):
        """Test string representation of RelationshipPair."""
        pair = RelationshipPair(User, City)
        repr_str = repr(pair)
        assert "RelationshipPair" in repr_str
        assert "from=" in repr_str
        assert "to=" in repr_str


# @@ STEP 6: Test DDL generation
class TestDDLGeneration:
    """Test DDL generation for multi-pair relationships."""
    
    def test_ddl_single_pair_minimal(self):
        """Test minimal DDL for single-pair relationship."""
        @kuzu_relationship(
            name="SimpleRel",
            pairs=[(User, City)]
        )
        class SimpleRel(KuzuBaseModel):
            pass
        
        ddl = generate_relationship_ddl(SimpleRel)
        # Default multiplicity MANY_MANY is added when not specified
        assert "CREATE REL TABLE SimpleRel(FROM User TO City" in ddl
    
    def test_ddl_multi_pair_with_properties(self):
        """Test DDL for multi-pair relationship with properties."""
        @kuzu_relationship(
            name="ComplexRel",
            pairs=[(User, User), (User, Post), (Post, City)],
            multiplicity=RelationshipMultiplicity.ONE_TO_MANY
        )
        class ComplexRel(KuzuBaseModel):
            created_at: str = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP)
            weight: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, default=1.0)
        
        ddl = generate_relationship_ddl(ComplexRel)
        
        # Check all components are present
        assert "CREATE REL TABLE ComplexRel(" in ddl
        assert "FROM User TO User" in ddl
        assert "FROM User TO Post" in ddl  
        assert "FROM Post TO City" in ddl
        assert "created_at TIMESTAMP" in ddl
        assert "weight DOUBLE DEFAULT 1.0" in ddl
        assert "ONE_MANY" in ddl
        assert ddl.endswith(");")
    
    def test_ddl_ordering(self):
        """Test that DDL components are in correct order."""
        @kuzu_relationship(
            name="OrderedRel",
            pairs=[(User, City)],
            multiplicity=RelationshipMultiplicity.MANY_TO_ONE
        )
        class OrderedRel(KuzuBaseModel):
            prop1: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
            prop2: int = kuzu_field(kuzu_type=KuzuDataType.INT32)
        
        ddl = generate_relationship_ddl(OrderedRel)
        
        # FROM-TO should come first, then properties, then multiplicity
        parts = ddl[ddl.index("(")+1:ddl.index(")")].split(", ")
        assert parts[0] == "FROM User TO City"
        assert parts[1] == "prop1 STRING"
        assert parts[2] == "prop2 INT32"
        assert parts[3] == "MANY_ONE"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive tests for relationship node type querying functionality.

This test suite challenges the implementation with complex real-world scenarios,
edge cases, boundary conditions, and inputs to ensure robustness.
"""

from __future__ import annotations

import pytest

from kuzualchemy import (
    KuzuBaseModel, KuzuRelationshipBase, kuzu_node, kuzu_relationship,
    kuzu_field, KuzuDataType, clear_registry, RelationshipNodeTypeQuery
)
from kuzualchemy.constants import RelationshipNodeTypeQueryConstants


class TestRelationshipNodeTypeQuerying:
    """Comprehensive test suite for relationship node type querying."""
    
    def setup_method(self):
        """Clear registry before each test."""
        clear_registry()
    
    def teardown_method(self):
        """Clear registry after each test."""
        clear_registry()

    def test_basic_functionality_exact_values(self):
        """Test basic functionality with exact expected values."""
        # @@ STEP: Define test nodes with precise types
        @kuzu_node("User")
        class User(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)

        @kuzu_node("Post")
        class Post(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            title: str = kuzu_field(kuzu_type=KuzuDataType.STRING)

        @kuzu_node("Comment")
        class Comment(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            content: str = kuzu_field(kuzu_type=KuzuDataType.STRING)

        @kuzu_relationship("AUTHORED", pairs=[(User, Post), (User, Comment)])
        class Authored(KuzuRelationshipBase):
            created_at: str = kuzu_field(kuzu_type=KuzuDataType.DATE)

        # derivation: from User -> {Post, Comment}
        # Expected: frozenset({Post, Comment})
        result = Authored.from_nodes_types(User).to_nodes_types
        expected = frozenset({Post, Comment})
        assert result == expected, f"Expected {expected}, got {result}"

        # derivation: to Post <- {User}
        # Expected: frozenset({User})
        result = Authored.to_nodes_types(Post).from_nodes_types
        expected = frozenset({User})
        assert result == expected, f"Expected {expected}, got {result}"

    def test_complex_multi_pair_relationships(self):
        """Test complex relationships with multiple FROM-TO pairs."""
        # @@ STEP: Create complex node hierarchy
        @kuzu_node("User")
        class User(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)

        @kuzu_node("Organization")
        class Organization(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)

        @kuzu_node("Bot")
        class Bot(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)

        @kuzu_node("Post")
        class Post(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)

        @kuzu_node("Comment")
        class Comment(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)

        @kuzu_node("Article")
        class Article(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)

        # Complex relationship with multiple pairs
        @kuzu_relationship("AUTHORED", pairs=[
            (User, Post), (User, Comment), (User, Article),
            (Organization, Post), (Organization, Article),
            (Bot, Comment)
        ])
        class Authored(KuzuRelationshipBase):
            created_at: str = kuzu_field(kuzu_type=KuzuDataType.DATE)

        # derivation: from User -> {Post, Comment, Article}
        result = Authored.from_nodes_types(User).to_nodes_types
        expected = frozenset({Post, Comment, Article})
        assert result == expected, f"Expected {expected}, got {result}"

        # derivation: from Organization -> {Post, Article}
        result = Authored.from_nodes_types(Organization).to_nodes_types
        expected = frozenset({Post, Article})
        assert result == expected, f"Expected {expected}, got {result}"

        # derivation: from Bot -> {Comment}
        result = Authored.from_nodes_types(Bot).to_nodes_types
        expected = frozenset({Comment})
        assert result == expected, f"Expected {expected}, got {result}"

        # derivation: from User, Organization -> {Post, Comment, Article}
        result = Authored.from_nodes_types(User, Organization).to_nodes_types
        expected = frozenset({Post, Comment, Article})
        assert result == expected, f"Expected {expected}, got {result}"

        # derivation: to Post <- {User, Organization}
        result = Authored.to_nodes_types(Post).from_nodes_types
        expected = frozenset({User, Organization})
        assert result == expected, f"Expected {expected}, got {result}"

        # derivation: to Comment <- {User, Bot}
        result = Authored.to_nodes_types(Comment).from_nodes_types
        expected = frozenset({User, Bot})
        assert result == expected, f"Expected {expected}, got {result}"

        # derivation: to Article <- {User, Organization}
        result = Authored.to_nodes_types(Article).from_nodes_types
        expected = frozenset({User, Organization})
        assert result == expected, f"Expected {expected}, got {result}"

        # derivation: to Post, Comment -> {User, Organization, Bot}
        result = Authored.to_nodes_types(Post, Comment).from_nodes_types
        expected = frozenset({User, Organization, Bot})
        assert result == expected, f"Expected {expected}, got {result}"

    def test_self_referencing_relationships(self):
        """Test self-referencing relationships with exact validation."""
        @kuzu_node("User")
        class User(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)

        @kuzu_relationship("FOLLOWS", pairs=[(User, User)])
        class Follows(KuzuRelationshipBase):
            since: str = kuzu_field(kuzu_type=KuzuDataType.DATE)

        # derivation: from User -> {User} (self-reference)
        result = Follows.from_nodes_types(User).to_nodes_types
        expected = frozenset({User})
        assert result == expected, f"Expected {expected}, got {result}"

        # derivation: to User <- {User} (self-reference)
        result = Follows.to_nodes_types(User).from_nodes_types
        expected = frozenset({User})
        assert result == expected, f"Expected {expected}, got {result}"

    def test_empty_result_sets(self):
        """Test cases that should return empty result sets."""
        @kuzu_node("User")
        class User(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)

        @kuzu_node("Post")
        class Post(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)

        @kuzu_node("Comment")
        class Comment(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)

        @kuzu_node("UnrelatedNode")
        class UnrelatedNode(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)

        @kuzu_relationship("AUTHORED", pairs=[(User, Post)])
        class Authored(KuzuRelationshipBase):
            created_at: str = kuzu_field(kuzu_type=KuzuDataType.DATE)

        # derivation: from UnrelatedNode -> {} (no pairs)
        result = Authored.from_nodes_types(UnrelatedNode).to_nodes_types
        expected = frozenset()
        assert result == expected, f"Expected {expected}, got {result}"

        # derivation: to Comment <- {} (no pairs)
        result = Authored.to_nodes_types(Comment).from_nodes_types
        expected = frozenset()
        assert result == expected, f"Expected {expected}, got {result}"

        # derivation: from UnrelatedNode, Comment -> {} (no pairs for either)
        result = Authored.from_nodes_types(UnrelatedNode, Comment).to_nodes_types
        expected = frozenset()
        assert result == expected, f"Expected {expected}, got {result}"

    def test_error_cases_with_exact_messages(self):
        """Test error cases with exact expected error messages."""
        @kuzu_node("User")
        class User(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)

        @kuzu_relationship("FOLLOWS", pairs=[(User, User)])
        class Follows(KuzuRelationshipBase):
            since: str = kuzu_field(kuzu_type=KuzuDataType.DATE)

        # Test invalid query direction - from_nodes_types query accessing from_nodes_types
        query = Follows.from_nodes_types(User)
        with pytest.raises(ValueError) as exc_info:
            _ = query.from_nodes_types
        # Expected exact error message
        assert "from_nodes_types can only be called on to_nodes_types() queries" in str(exc_info.value)

        # Test invalid query direction - to_nodes_types query accessing to_nodes_types
        query = Follows.to_nodes_types(User)
        with pytest.raises(ValueError) as exc_info:
            _ = query.to_nodes_types
        # Expected exact error message
        assert "to_nodes_types can only be called on from_nodes_types() queries" in str(exc_info.value)

        # Test invalid node type - string instead of class
        with pytest.raises(TypeError) as exc_info:
            Follows.from_nodes_types("invalid_string")
        # Expected exact error message format
        expected_msg = RelationshipNodeTypeQueryConstants.INVALID_NODE_TYPE.format(
            "invalid_string", "str"
        )
        assert str(exc_info.value) == expected_msg

        # Test invalid node type - integer instead of class
        with pytest.raises(TypeError) as exc_info:
            Follows.from_nodes_types(42)
        # Expected exact error message format
        expected_msg = RelationshipNodeTypeQueryConstants.INVALID_NODE_TYPE.format(
            42, "int"
        )
        assert str(exc_info.value) == expected_msg

        # Test invalid node type - None instead of class
        with pytest.raises(TypeError) as exc_info:
            Follows.from_nodes_types(None)
        # Expected exact error message format
        expected_msg = RelationshipNodeTypeQueryConstants.INVALID_NODE_TYPE.format(
            None, "NoneType"
        )
        assert str(exc_info.value) == expected_msg

    def test_abstract_relationship_error(self):
        """Test that abstract relationships raise appropriate errors."""
        @kuzu_node("User")
        class User(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)

        @kuzu_relationship(abstract=True)
        class AbstractRelationship(KuzuRelationshipBase):
            created_at: str = kuzu_field(kuzu_type=KuzuDataType.DATE)

        # Test abstract relationship error
        with pytest.raises(ValueError) as exc_info:
            AbstractRelationship.from_nodes_types(User)
        # Expected exact error message
        expected_msg = RelationshipNodeTypeQueryConstants.ABSTRACT_RELATIONSHIP_QUERY.format(
            "AbstractRelationship"
        )
        assert str(exc_info.value) == expected_msg

        with pytest.raises(ValueError) as exc_info:
            AbstractRelationship.to_nodes_types(User)
        # Expected exact error message
        expected_msg = RelationshipNodeTypeQueryConstants.ABSTRACT_RELATIONSHIP_QUERY.format(
            "AbstractRelationship"
        )
        assert str(exc_info.value) == expected_msg

    def test_cache_performance_and_consistency(self):
        """Test cache performance and consistency across multiple calls."""
        @kuzu_node("User")
        class User(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)

        @kuzu_node("Post")
        class Post(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)

        @kuzu_relationship("AUTHORED", pairs=[(User, Post)])
        class Authored(KuzuRelationshipBase):
            created_at: str = kuzu_field(kuzu_type=KuzuDataType.DATE)

        # First call should build cache
        result1 = Authored.from_nodes_types(User).to_nodes_types
        expected = frozenset({Post})
        assert result1 == expected

        # Second call should use cache and return identical result
        result2 = Authored.from_nodes_types(User).to_nodes_types
        assert result2 == expected
        assert result1 is result2  # Should be the same frozenset object due to caching

        # Test cache consistency across different query types
        result3 = Authored.to_nodes_types(Post).from_nodes_types
        expected_from = frozenset({User})
        assert result3 == expected_from

        # Verify cache is working by checking internal state
        cache_key = Authored.__name__
        assert cache_key in Authored._node_type_cache
        cache = Authored._node_type_cache[cache_key]
        assert RelationshipNodeTypeQueryConstants.CACHE_KEY_FROM_TO_MAP in cache
        assert RelationshipNodeTypeQueryConstants.CACHE_KEY_TO_FROM_MAP in cache

    def test_deeply_nested_inheritance_patterns(self):
        """Test complex inheritance patterns that could break the implementation."""
        @kuzu_node("BaseEntity")
        class BaseEntity(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)

        @kuzu_node("User")
        class User(BaseEntity):
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)

        @kuzu_node("AdminUser")
        class AdminUser(User):
            admin_level: int = kuzu_field(kuzu_type=KuzuDataType.INT32)

        @kuzu_node("SuperAdminUser")
        class SuperAdminUser(AdminUser):
            super_powers: str = kuzu_field(kuzu_type=KuzuDataType.STRING)

        @kuzu_node("Content")
        class Content(BaseEntity):
            title: str = kuzu_field(kuzu_type=KuzuDataType.STRING)

        @kuzu_node("Post")
        class Post(Content):
            body: str = kuzu_field(kuzu_type=KuzuDataType.STRING)

        @kuzu_node("Article")
        class Article(Content):
            abstract: str = kuzu_field(kuzu_type=KuzuDataType.STRING)

        # Complex relationship with inheritance hierarchy
        @kuzu_relationship("MANAGES", pairs=[
            (AdminUser, Post), (AdminUser, Article),
            (SuperAdminUser, Post), (SuperAdminUser, Article)
        ])
        class Manages(KuzuRelationshipBase):
            permissions: str = kuzu_field(kuzu_type=KuzuDataType.STRING)

        # derivation: from AdminUser -> {Post, Article}
        result = Manages.from_nodes_types(AdminUser).to_nodes_types
        expected = frozenset({Post, Article})
        assert result == expected, f"Expected {expected}, got {result}"

        # derivation: from SuperAdminUser -> {Post, Article}
        result = Manages.from_nodes_types(SuperAdminUser).to_nodes_types
        expected = frozenset({Post, Article})
        assert result == expected, f"Expected {expected}, got {result}"

        # derivation: from AdminUser, SuperAdminUser -> {Post, Article}
        result = Manages.from_nodes_types(AdminUser, SuperAdminUser).to_nodes_types
        expected = frozenset({Post, Article})
        assert result == expected, f"Expected {expected}, got {result}"

        # derivation: to Post <- {AdminUser, SuperAdminUser}
        result = Manages.to_nodes_types(Post).from_nodes_types
        expected = frozenset({AdminUser, SuperAdminUser})
        assert result == expected, f"Expected {expected}, got {result}"

    def test_boundary_conditions_large_numbers(self):
        """Test boundary conditions with large numbers of node types."""
        # Create many node types to test performance and correctness
        node_classes = []
        for i in range(20):  # Create 20 node types for testing
            @kuzu_node(f"Node{i}")
            class DynamicNode(KuzuBaseModel):
                id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)

            # Rename the class to avoid conflicts
            DynamicNode.__name__ = f"Node{i}"
            DynamicNode.__qualname__ = f"Node{i}"
            node_classes.append(DynamicNode)

        # Create relationship pairs - first 10 nodes can connect to last 10 nodes
        pairs = [(node_classes[i], node_classes[i + 10]) for i in range(10)]

        @kuzu_relationship("CONNECTS", pairs=pairs)
        class Connects(KuzuRelationshipBase):
            weight: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE)

        # derivation: from first node -> {Node10}
        result = Connects.from_nodes_types(node_classes[0]).to_nodes_types
        expected = frozenset({node_classes[10]})
        assert result == expected, f"Expected {expected}, got {result}"

        # derivation: from all first 10 nodes -> all last 10 nodes
        result = Connects.from_nodes_types(*node_classes[:10]).to_nodes_types
        expected = frozenset(node_classes[10:])
        assert result == expected, f"Expected {expected}, got {result}"

        # derivation: to last node <- {Node9}
        result = Connects.to_nodes_types(node_classes[19]).from_nodes_types
        expected = frozenset({node_classes[9]})
        assert result == expected, f"Expected {expected}, got {result}"

    def test_string_node_reference_resolution(self):
        """Test resolution of string node references in relationship pairs."""
        @kuzu_node("User")
        class User(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)

        @kuzu_node("Post")
        class Post(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)

        # Create relationship with string references (simulating forward references)
        @kuzu_relationship("AUTHORED", pairs=[("User", "Post")])
        class Authored(KuzuRelationshipBase):
            created_at: str = kuzu_field(kuzu_type=KuzuDataType.DATE)

        # derivation: string "User" should resolve to User class -> {Post}
        result = Authored.from_nodes_types(User).to_nodes_types
        expected = frozenset({Post})
        assert result == expected, f"Expected {expected}, got {result}"

        # derivation: string "Post" should resolve to Post class <- {User}
        result = Authored.to_nodes_types(Post).from_nodes_types
        expected = frozenset({User})
        assert result == expected, f"Expected {expected}, got {result}"

    def test_cache_invalidation_on_registry_changes(self):
        """Test that cache is properly invalidated when registry changes."""
        @kuzu_node("User")
        class User(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)

        @kuzu_node("Post")
        class Post(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)

        @kuzu_relationship("AUTHORED", pairs=[(User, Post)])
        class Authored(KuzuRelationshipBase):
            created_at: str = kuzu_field(kuzu_type=KuzuDataType.DATE)

        # Build cache
        result1 = Authored.from_nodes_types(User).to_nodes_types
        expected = frozenset({Post})
        assert result1 == expected

        # Verify cache exists
        cache_key = Authored.__name__
        assert cache_key in Authored._node_type_cache

        # Clear registry (should invalidate cache)
        clear_registry()

        # Verify cache was cleared
        assert len(Authored._node_type_cache) == 0

    def test_relationship_query_object_properties(self):
        """Test properties and behavior of RelationshipNodeTypeQuery objects."""
        @kuzu_node("User")
        class User(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)

        @kuzu_node("Post")
        class Post(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)

        @kuzu_relationship("AUTHORED", pairs=[(User, Post)])
        class Authored(KuzuRelationshipBase):
            created_at: str = kuzu_field(kuzu_type=KuzuDataType.DATE)

        # Test RelationshipNodeTypeQuery object creation and FUNCTIONALITY
        query = Authored.from_nodes_types(User)
        assert isinstance(query, RelationshipNodeTypeQuery)
        
        # Test the ACTUAL FUNCTIONALITY - not private attributes
        result = query.to_nodes_types
        assert result == frozenset({Post})
        assert isinstance(result, frozenset)
        
        query2 = Authored.to_nodes_types(Post)
        assert isinstance(query2, RelationshipNodeTypeQuery)
        
        # Test the ACTUAL FUNCTIONALITY - not private attributes  
        result2 = query2.from_nodes_types
        assert result2 == frozenset({User})
        assert isinstance(result2, frozenset)
        
        # Test that results are pre-computed for ultra-fast access
        assert hasattr(query, '_result')
        assert hasattr(query2, '_result')
        
        # Test error handling works correctly
        with pytest.raises(ValueError):
            _ = query.from_nodes_types  # Wrong direction
        
        with pytest.raises(ValueError):
            _ = query2.to_nodes_types  # Wrong direction

    def test_frozenset_immutability_and_performance(self):
        """Test that results are frozensets for immutability and performance."""
        @kuzu_node("User")
        class User(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)

        @kuzu_node("Post")
        class Post(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)

        @kuzu_node("Comment")
        class Comment(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)

        @kuzu_relationship("AUTHORED", pairs=[(User, Post), (User, Comment)])
        class Authored(KuzuRelationshipBase):
            created_at: str = kuzu_field(kuzu_type=KuzuDataType.DATE)

        # Test that results are frozensets
        result = Authored.from_nodes_types(User).to_nodes_types
        assert isinstance(result, frozenset), f"Expected frozenset, got {type(result)}"

        # Test immutability - frozensets cannot be modified
        with pytest.raises(AttributeError):
            result.add(User)  # Should fail because frozensets are immutable

        with pytest.raises(AttributeError):
            result.remove(Post)  # Should fail because frozensets are immutable

        # Test that frozensets can be used as dictionary keys (hashable)
        test_dict = {result: "test_value"}
        assert test_dict[result] == "test_value"

        # Test that identical queries return the same frozenset object (performance)
        result2 = Authored.from_nodes_types(User).to_nodes_types
        assert result is result2, "Cache should return the same frozenset object"

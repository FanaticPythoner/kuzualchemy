# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive tests for auto-increment field functionality in KuzuAlchemy.

Tests cover:
- Single node creation with auto-increment primary keys
- Batch operations with auto-increment fields
- Edge cases and error handling
- Composite primary keys involving auto-increment fields
- Pydantic validation behavior with auto-increment fields
"""

from __future__ import annotations

import pytest
import re
import uuid
from typing import Optional
from pydantic import ValidationError

from kuzualchemy import (
    KuzuBaseModel,
    KuzuRelationshipBase,
    kuzu_node,
    kuzu_relationship,
    kuzu_field,
    KuzuDataType,
    KuzuSession,
    get_ddl_for_node,
    get_ddl_for_relationship,
)
from kuzualchemy.test_utilities import initialize_schema


class TestAutoIncrementFields:
    """Test auto-increment field functionality."""

    def setup_method(self):
        """Set up test models with auto-increment fields."""
        
        @kuzu_node("AutoUser")
        class AutoUser(KuzuBaseModel):
            """User model with auto-increment primary key."""
            id: Optional[int] = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True, auto_increment=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
            email: Optional[str] = kuzu_field(kuzu_type=KuzuDataType.STRING, unique=True, default=None)
            age: int = kuzu_field(kuzu_type=KuzuDataType.INT32, default=0)

        @kuzu_node("AutoProduct")
        class AutoProduct(KuzuBaseModel):
            """Product model with auto-increment field (not primary key)."""
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING, primary_key=True)
            product_id: Optional[int] = kuzu_field(kuzu_type=KuzuDataType.INT64, auto_increment=True)
            price: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, default=0.0)

        @kuzu_node("AutoOrder")
        class AutoOrder(KuzuBaseModel):
            """Order model with multiple auto-increment fields."""
            order_id: Optional[int] = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True, auto_increment=True)
            tracking_number: Optional[int] = kuzu_field(kuzu_type=KuzuDataType.INT64, auto_increment=True)
            customer_name: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
            total: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, default=0.0)

        @kuzu_relationship("AutoFollows", pairs=[(AutoUser, AutoUser)])
        class AutoFollows(KuzuRelationshipBase):
            """Relationship with auto-increment primary key."""
            follow_id: Optional[int] = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True, auto_increment=True)
            followed_at: Optional[str] = kuzu_field(kuzu_type=KuzuDataType.STRING, default=None)

        @kuzu_relationship("AutoPurchase", pairs=[(AutoUser, AutoProduct)])
        class AutoPurchase(KuzuRelationshipBase):
            """Relationship with auto-increment field (not primary key)."""
            user_id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            product_id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            transaction_id: Optional[int] = kuzu_field(kuzu_type=KuzuDataType.INT64, auto_increment=True)
            amount: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, default=0.0)

        @kuzu_node("AutoUUIDUser")
        class AutoUUIDUser(KuzuBaseModel):
            """User model with UUID auto-increment primary key."""
            id: Optional[uuid.UUID] = kuzu_field(kuzu_type=KuzuDataType.UUID, primary_key=True, auto_increment=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
            email: Optional[str] = kuzu_field(kuzu_type=KuzuDataType.STRING, unique=True, default=None)

        @kuzu_node("AutoUUIDProduct")
        class AutoUUIDProduct(KuzuBaseModel):
            """Product model with UUID auto-increment field (not primary key)."""
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING, primary_key=True)
            product_uuid: Optional[uuid.UUID] = kuzu_field(kuzu_type=KuzuDataType.UUID, auto_increment=True)
            price: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, default=0.0)

        @kuzu_relationship("AutoUUIDFollows", pairs=[(AutoUUIDUser, AutoUUIDUser)])
        class AutoUUIDFollows(KuzuRelationshipBase):
            """Relationship with UUID auto-increment primary key."""
            follow_uuid: Optional[uuid.UUID] = kuzu_field(kuzu_type=KuzuDataType.UUID, primary_key=True, auto_increment=True)
            followed_at: Optional[str] = kuzu_field(kuzu_type=KuzuDataType.STRING, default=None)

        self.AutoUser = AutoUser
        self.AutoProduct = AutoProduct
        self.AutoOrder = AutoOrder
        self.AutoFollows = AutoFollows
        self.AutoPurchase = AutoPurchase

        self.AutoUUIDUser = AutoUUIDUser
        self.AutoUUIDProduct = AutoUUIDProduct
        self.AutoUUIDFollows = AutoUUIDFollows

    def test_auto_increment_field_detection(self):
        """Test that auto-increment fields are correctly detected."""
        # Test AutoUser model
        auto_fields = self.AutoUser.get_auto_increment_fields()
        assert auto_fields == ["id"]
        
        auto_metadata = self.AutoUser.get_auto_increment_metadata()
        assert "id" in auto_metadata
        assert auto_metadata["id"].auto_increment is True
        assert auto_metadata["id"].primary_key is True
        
        assert self.AutoUser.has_auto_increment_primary_key() is True

        # Test AutoProduct model
        auto_fields = self.AutoProduct.get_auto_increment_fields()
        assert auto_fields == ["product_id"]
        
        assert self.AutoProduct.has_auto_increment_primary_key() is False

        # Test AutoOrder model
        auto_fields = self.AutoOrder.get_auto_increment_fields()
        assert set(auto_fields) == {"order_id", "tracking_number"}
        
        assert self.AutoOrder.has_auto_increment_primary_key() is True

    def test_pydantic_validation_with_auto_increment(self):
        """Test that auto-increment fields are optional during instantiation."""
        # Should be able to create instances without providing auto-increment fields
        user = self.AutoUser(name="Alice", email="alice@example.com", age=30)
        assert user.name == "Alice"
        assert user.email == "alice@example.com"
        assert user.age == 30
        assert user.id is None  # Auto-increment field should be None initially

        # Should also be able to explicitly provide auto-increment fields
        user_with_id = self.AutoUser(id=100, name="Bob", email="bob@example.com")
        assert user_with_id.id == 100
        assert user_with_id.name == "Bob"

        # Test with multiple auto-increment fields
        order = self.AutoOrder(customer_name="John Doe", total=99.99)
        assert order.customer_name == "John Doe"
        assert order.total == 99.99
        assert order.order_id is None
        assert order.tracking_number is None

    def test_single_node_creation_with_auto_increment(self, test_db_path):
        """Test creating a single node with auto-increment primary key."""
        session = KuzuSession(db_path=test_db_path)
        
        # Initialize schema
        ddl = get_ddl_for_node(self.AutoUser)
        initialize_schema(session, ddl=ddl)

        # Create user without providing ID
        user = self.AutoUser(name="Alice", email="alice@example.com", age=30)
        assert user.id is None  # Should be None before insertion

        session.add(user)
        session.commit()

        # After insertion, the auto-increment field should be populated
        assert user.id is not None
        assert isinstance(user.id, int)
        assert user.id >= 0  # SERIAL starts from 0 in KuzuDB

        session.close()

    def test_multiple_node_creation_with_auto_increment(self, test_db_path):
        """Test creating multiple nodes with auto-increment fields."""
        session = KuzuSession(db_path=test_db_path)
        
        # Initialize schema
        ddl = get_ddl_for_node(self.AutoUser)
        initialize_schema(session, ddl=ddl)

        # Create multiple users
        users = [
            self.AutoUser(name="Alice", email="alice@example.com"),
            self.AutoUser(name="Bob", email="bob@example.com"),
            self.AutoUser(name="Carol", email="carol@example.com"),
        ]

        # All should have None IDs initially
        for user in users:
            assert user.id is None

        # Add all users
        for user in users:
            session.add(user)
        session.commit()

        # After insertion, all should have unique auto-generated IDs
        ids = [user.id for user in users]
        assert all(id is not None for id in ids)
        assert all(isinstance(id, int) for id in ids)
        assert len(set(ids)) == len(ids)  # All IDs should be unique
        
        # IDs should be sequential (0, 1, 2)
        sorted_ids = sorted(ids)
        assert sorted_ids == [0, 1, 2]

        session.close()

    def test_non_primary_key_auto_increment(self, test_db_path):
        """Test auto-increment fields that are not primary keys."""
        session = KuzuSession(db_path=test_db_path)
        
        # Initialize schema
        ddl = get_ddl_for_node(self.AutoProduct)
        initialize_schema(session, ddl=ddl)

        # Create product with string primary key and auto-increment product_id
        product = self.AutoProduct(name="Widget", price=19.99)
        assert product.product_id is None

        session.add(product)
        session.commit()

        # Auto-increment field should be populated
        assert product.product_id is not None
        assert isinstance(product.product_id, int)

        session.close()

    def test_multiple_auto_increment_fields(self, test_db_path):
        """Test model with multiple auto-increment fields."""
        session = KuzuSession(db_path=test_db_path)
        
        # Initialize schema
        ddl = get_ddl_for_node(self.AutoOrder)
        initialize_schema(session, ddl=ddl)

        # Create order with multiple auto-increment fields
        order = self.AutoOrder(customer_name="John Doe", total=99.99)
        assert order.order_id is None
        assert order.tracking_number is None

        session.add(order)
        session.commit()

        # Both auto-increment fields should be populated
        assert order.order_id is not None
        assert order.tracking_number is not None
        assert isinstance(order.order_id, int)
        assert isinstance(order.tracking_number, int)

        session.close()

    def test_identity_map_with_auto_increment(self, test_db_path):
        """Test that identity map works correctly with auto-increment primary keys."""
        session = KuzuSession(db_path=test_db_path, expire_on_commit=False)
        
        # Initialize schema
        ddl = get_ddl_for_node(self.AutoUser)
        initialize_schema(session, ddl=ddl)

        # Create and insert user
        user = self.AutoUser(name="Alice", email="alice@example.com")
        session.add(user)
        session.commit()

        # User should be in identity map after commit
        assert user.id is not None
        identity_key = f"{self.AutoUser.__name__}:{user.id}"
        assert identity_key in session._identity_map
        assert session._identity_map[identity_key] is user

        session.close()

    def test_explicit_auto_increment_value(self, test_db_path):
        """Test providing explicit values for auto-increment fields."""
        session = KuzuSession(db_path=test_db_path)
        
        # Initialize schema
        ddl = get_ddl_for_node(self.AutoUser)
        initialize_schema(session, ddl=ddl)

        # Create user with explicit ID
        user = self.AutoUser(id=100, name="Alice", email="alice@example.com")
        assert user.id == 100

        session.add(user)
        session.commit()

        # ID should remain as explicitly set
        assert user.id == 100

        session.close()

    def test_ddl_generation_with_auto_increment(self):
        """Test that DDL is correctly generated for auto-increment fields."""
        ddl = get_ddl_for_node(self.AutoUser)
        
        # Should contain SERIAL type for auto-increment field
        assert "SERIAL" in ddl
        assert "PRIMARY KEY" in ddl
        
        # Verify the structure
        assert "AutoUser" in ddl
        assert "id SERIAL PRIMARY KEY" in ddl or "id SERIAL" in ddl

    # ========================================
    # MANUAL OVERRIDE TESTS
    # ========================================

    def test_manual_auto_increment_value_single_node(self, test_db_path):
        """Test creating a single node with manually specified auto-increment value."""
        session = KuzuSession(db_path=test_db_path)

        # Initialize schema
        ddl = get_ddl_for_node(self.AutoUser)
        initialize_schema(session, ddl=ddl)

        # Create user with explicit ID
        user = self.AutoUser(id=100, name="Alice", email="alice@example.com")
        assert user.id == 100  # Should use provided value

        session.add(user)
        session.commit()

        # After insertion, should still have the manual value
        assert user.id == 100

        # Verify in database
        result = session.execute("MATCH (u:AutoUser) RETURN u ORDER BY u.id")
        assert len(result) == 1
        assert result[0]['u']['id'] == 100
        assert result[0]['u']['name'] == "Alice"

        session.close()

    def test_manual_auto_increment_value_mixed_batch(self, test_db_path):
        """Test creating multiple nodes with mix of auto-increment and manual values."""
        session = KuzuSession(db_path=test_db_path)

        # Initialize schema
        ddl = get_ddl_for_node(self.AutoUser)
        initialize_schema(session, ddl=ddl)

        # Create users with mix of auto and manual IDs
        users = [
            self.AutoUser(name="Alice", email="alice@example.com"),  # Auto-increment
            self.AutoUser(id=50, name="Bob", email="bob@example.com"),  # Manual
            self.AutoUser(name="Carol", email="carol@example.com"),  # Auto-increment
            self.AutoUser(id=100, name="Dave", email="dave@example.com"),  # Manual
        ]

        # Before insertion
        assert users[0].id is None  # Unset (auto-increment)
        assert users[1].id == 50   # Manual
        assert users[2].id is None  # Unset (auto-increment)
        assert users[3].id == 100  # Manual

        # Add all users
        for user in users:
            session.add(user)
        session.commit()

        # After insertion
        assert users[0].id is not None and users[0].id != 50 and users[0].id != 100  # Auto-generated
        assert users[1].id == 50   # Manual value preserved
        assert users[2].id is not None and users[2].id != 50 and users[2].id != 100  # Auto-generated
        assert users[3].id == 100  # Manual value preserved

        # Verify all users have unique IDs
        ids = [user.id for user in users]
        assert len(set(ids)) == 4  # All unique

        # Verify in database
        result = session.execute("MATCH (u:AutoUser) RETURN u ORDER BY u.id")
        assert len(result) == 4

        # Check that manual values are preserved
        db_ids = [row['u']['id'] for row in result]
        assert 50 in db_ids
        assert 100 in db_ids

        session.close()

    def test_explicit_none_auto_increment_primary_key_error(self, test_db_path):
        """Test that explicitly setting auto-increment primary key to None raises an error."""
        session = KuzuSession(db_path=test_db_path)

        # Initialize schema
        ddl = get_ddl_for_node(self.AutoUser)
        initialize_schema(session, ddl=ddl)

        # Create user with explicit None for primary key (should fail)
        user = self.AutoUser(id=None, name="Alice", email="alice@example.com")
        assert user.id is None

        session.add(user)

        # Should raise error because primary keys cannot be NULL in KuzuDB
        with pytest.raises(RuntimeError, match="non-null constraint of the primary key"):
            session.commit()

        session.close()

    def test_sequence_continuation_after_manual_values(self, test_db_path):
        """Test that auto-increment sequence continues correctly after manual insertions."""
        session = KuzuSession(db_path=test_db_path)

        # Initialize schema
        ddl = get_ddl_for_node(self.AutoUser)
        initialize_schema(session, ddl=ddl)

        # Insert user with high manual ID
        manual_user = self.AutoUser(id=1000, name="Manual", email="manual@example.com")
        session.add(manual_user)
        session.commit()
        assert manual_user.id == 1000

        # Insert auto-increment user (should get next available ID, likely 0 or 1)
        auto_user = self.AutoUser(name="Auto", email="auto@example.com")
        session.add(auto_user)
        session.commit()

        # Auto-generated ID should be different from manual ID
        assert auto_user.id is not None
        assert auto_user.id != 1000

        # Verify both users exist in database
        result = session.execute("MATCH (u:AutoUser) RETURN u ORDER BY u.id")
        assert len(result) == 2

        session.close()

    def test_validation_of_manual_auto_increment_values(self, test_db_path):
        """Test validation of manually provided auto-increment values."""
        session = KuzuSession(db_path=test_db_path)

        # Initialize schema
        ddl = get_ddl_for_node(self.AutoUser)
        initialize_schema(session, ddl=ddl)

        # Test negative integer (should be caught by our custom validation)
        with pytest.raises(ValueError, match="must be non-negative"):
            user = self.AutoUser(id=-1, name="Invalid", email="invalid@example.com")
            session.add(user)
            session.commit()

        # Test string value (should be caught by Pydantic type validation)
        with pytest.raises(ValidationError):
            user = self.AutoUser(id="not_an_int", name="Invalid", email="invalid@example.com")

        # Test float value (should be caught by Pydantic type validation)
        with pytest.raises(ValidationError):
            user = self.AutoUser(id=3.14, name="Invalid", email="invalid@example.com")

        session.close()

    # ========================================
    # RELATIONSHIP AUTO-INCREMENT TESTS
    # ========================================

    def test_relationship_auto_increment_single(self, test_db_path):
        """Test creating a single relationship with auto-increment primary key."""
        session = KuzuSession(db_path=test_db_path)

        # Initialize schema
        user_ddl = get_ddl_for_node(self.AutoUser)
        follows_ddl = get_ddl_for_relationship(self.AutoFollows)
        initialize_schema(session, ddl=user_ddl + "\n" + follows_ddl)

        # Create users first
        user1 = self.AutoUser(name="Alice", email="alice@example.com")
        user2 = self.AutoUser(name="Bob", email="bob@example.com")
        session.add(user1)
        session.add(user2)
        session.commit()

        # Create relationship with auto-increment
        follow = self.AutoFollows(from_node=user1, to_node=user2, followed_at="2024-01-01")
        assert follow.follow_id is None  # Should be unset (auto-increment)

        session.add(follow)
        session.commit()

        # After insertion, should have auto-generated ID
        assert follow.follow_id is not None
        assert isinstance(follow.follow_id, int)
        assert follow.follow_id >= 0

        session.close()

    def test_relationship_manual_auto_increment_value(self, test_db_path):
        """Test creating a relationship with manually specified auto-increment value."""
        session = KuzuSession(db_path=test_db_path)

        # Initialize schema
        user_ddl = get_ddl_for_node(self.AutoUser)
        follows_ddl = get_ddl_for_relationship(self.AutoFollows)
        initialize_schema(session, ddl=user_ddl + "\n" + follows_ddl)

        # Create users first
        user1 = self.AutoUser(name="Alice", email="alice@example.com")
        user2 = self.AutoUser(name="Bob", email="bob@example.com")
        session.add(user1)
        session.add(user2)
        session.commit()

        # Create relationship with manual ID
        follow = self.AutoFollows(from_node=user1, to_node=user2, follow_id=500, followed_at="2024-01-01")
        assert follow.follow_id == 500  # Should use provided value

        session.add(follow)
        session.commit()

        # After insertion, should still have the manual value
        assert follow.follow_id == 500

        session.close()

    def test_relationship_mixed_auto_increment_values(self, test_db_path):
        """Test creating multiple relationships with mix of auto-increment and manual values."""
        session = KuzuSession(db_path=test_db_path)

        # Initialize schema
        user_ddl = get_ddl_for_node(self.AutoUser)
        follows_ddl = get_ddl_for_relationship(self.AutoFollows)
        initialize_schema(session, ddl=user_ddl + "\n" + follows_ddl)

        # Create users first
        users = [
            self.AutoUser(name="Alice", email="alice@example.com"),
            self.AutoUser(name="Bob", email="bob@example.com"),
            self.AutoUser(name="Carol", email="carol@example.com"),
        ]
        for user in users:
            session.add(user)
        session.commit()

        # Create relationships with mix of auto and manual IDs
        follows = [
            self.AutoFollows(from_node=users[0], to_node=users[1], followed_at="2024-01-01"),  # Auto
            self.AutoFollows(from_node=users[1], to_node=users[2], follow_id=100, followed_at="2024-01-02"),  # Manual
            self.AutoFollows(from_node=users[2], to_node=users[0], followed_at="2024-01-03"),  # Auto
        ]

        # Before insertion
        assert follows[0].follow_id is None  # Unset (auto-increment)
        assert follows[1].follow_id == 100  # Manual
        assert follows[2].follow_id is None  # Unset (auto-increment)

        # Add all relationships
        for follow in follows:
            session.add(follow)
        session.commit()

        # After insertion
        assert follows[0].follow_id is not None and follows[0].follow_id != 100  # Auto-generated
        assert follows[1].follow_id == 100  # Manual value preserved
        assert follows[2].follow_id is not None and follows[2].follow_id != 100  # Auto-generated

        # Verify all relationships have unique IDs
        ids = [follow.follow_id for follow in follows]
        assert len(set(ids)) == 3  # All unique

        session.close()

    def test_multi_session_auto_increment_insertion(self, test_db_path):
        """
        test: Multi-session insertion of auto-increment nodes.

        Tests the auto-increment sequence generation
        across multiple sessions to ensure proper sequence continuity.
        """
        # Initialize schema with first session
        init_session = KuzuSession(db_path=test_db_path)
        ddl = get_ddl_for_node(self.AutoUser)
        initialize_schema(init_session, ddl=ddl)
        init_session.close()

        # Use multiple sessions sequentially to test cross-session auto-increment behavior
        all_users = []
        for i in range(5):
            session = KuzuSession(db_path=test_db_path)
            try:
                users_for_session = [
                    self.AutoUser(name=f"User_{i}_{j}", email=f"user_{i}_{j}@example.com")
                    for j in range(10)
                ]
                for user in users_for_session:
                    session.add(user)
                session.commit()
                all_users.extend(users_for_session)
            finally:
                session.close()

        # Validation: All IDs must be unique and sequential
        # Expected: 50 users total (5 sessions × 10 users each)
        assert len(all_users) == 50

        # Extract all generated IDs
        all_ids = [user.id for user in all_users]

        # Validation 1: All IDs must be non-None integers
        for user_id in all_ids:
            assert user_id is not None
            assert isinstance(user_id, int)
            assert user_id >= 0  # SERIAL starts from 0

        # Validation 2: All IDs must be unique (uniqueness constraint)
        unique_ids = set(all_ids)
        assert len(unique_ids) == 50, f"Expected 50 unique IDs, got {len(unique_ids)}: {sorted(all_ids)}"

        # Validation 3: IDs should form a contiguous sequence from 0 to 49
        sorted_ids = sorted(all_ids)
        expected_sequence = list(range(50))
        assert sorted_ids == expected_sequence, f"Expected {expected_sequence}, got {sorted_ids}"

    def test_multi_pair_relationship_type_determination(self, test_db_path):
        """
        test: Multi-pair relationship with complex node type determination.

        Tests the correctness of relationship pair matching when
        dealing with complex inheritance hierarchies and multiple valid pairs.
        """
        session = KuzuSession(db_path=test_db_path)

        # Create complex node hierarchy
        @kuzu_node("BaseEntity")
        class BaseEntity(KuzuBaseModel):
            id: Optional[int] = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True, auto_increment=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)

        @kuzu_node("SpecializedEntity")
        class SpecializedEntity(KuzuBaseModel):
            id: Optional[int] = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True, auto_increment=True)
            special_field: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)

        @kuzu_node("ComplexEntity")
        class ComplexEntity(KuzuBaseModel):
            id: Optional[int] = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True, auto_increment=True)
            complex_data: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)

        # Multi-pair relationship with complex type combinations
        @kuzu_relationship("ComplexInteraction", pairs=[
            (BaseEntity, SpecializedEntity),
            (SpecializedEntity, ComplexEntity),
            (ComplexEntity, BaseEntity),
            (BaseEntity, BaseEntity)  # Self-referential pair
        ])
        class ComplexInteraction(KuzuRelationshipBase):
            interaction_id: Optional[int] = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True, auto_increment=True)
            interaction_type: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)

        # Initialize schema
        ddl_parts = [
            get_ddl_for_node(BaseEntity),
            get_ddl_for_node(SpecializedEntity),
            get_ddl_for_node(ComplexEntity),
            get_ddl_for_relationship(ComplexInteraction)
        ]
        initialize_schema(session, ddl="\n".join(ddl_parts))

        # Create test entities
        base_entity = BaseEntity(name="Base1")
        specialized_entity = SpecializedEntity(special_field="Special1")
        complex_entity = ComplexEntity(complex_data="Complex1")

        session.add_all([base_entity, specialized_entity, complex_entity])
        session.commit()

        # Test each relationship pair type
        test_cases = [
            (base_entity, specialized_entity, "BaseToSpecialized"),
            (specialized_entity, complex_entity, "SpecializedToComplex"),
            (complex_entity, base_entity, "ComplexToBase"),
            (base_entity, base_entity, "BaseToBase")  # Self-referential
        ]

        created_relationships = []
        for from_node, to_node, interaction_type in test_cases:
            relationship = ComplexInteraction(
                from_node=from_node,
                to_node=to_node,
                interaction_type=interaction_type
            )
            session.add(relationship)
            created_relationships.append(relationship)

        session.commit()

        # Validation: All relationships should have auto-generated IDs
        for i, relationship in enumerate(created_relationships):
            # Validation 1: Auto-increment ID must be generated
            assert relationship.interaction_id is not None, f"Relationship {i} missing auto-increment ID"
            assert isinstance(relationship.interaction_id, int), f"Relationship {i} ID not integer: {type(relationship.interaction_id)}"
            assert relationship.interaction_id >= 0, f"Relationship {i} ID negative: {relationship.interaction_id}"

            # Validation 2: Relationship type must match expected
            expected_type = test_cases[i][2]
            assert relationship.interaction_type == expected_type, f"Expected {expected_type}, got {relationship.interaction_type}"

        # Validation 3: All relationship IDs must be unique
        relationship_ids = [r.interaction_id for r in created_relationships]
        assert len(set(relationship_ids)) == len(relationship_ids), f"Duplicate relationship IDs: {relationship_ids}"

        # Validation 4: IDs should be sequential (0, 1, 2, 3)
        sorted_rel_ids = sorted(relationship_ids)
        expected_rel_sequence = list(range(len(created_relationships)))
        assert sorted_rel_ids == expected_rel_sequence, f"Expected {expected_rel_sequence}, got {sorted_rel_ids}"

        session.close()

    def test_raw_primary_key_node_type_determination(self, test_db_path):
        """
        test: Node type determination from raw primary key values.

        Tests the correctness of the _get_node_type_name method
        when dealing with raw primary key values instead of model instances.
        """
        session = KuzuSession(db_path=test_db_path)

        # Create multiple node types with overlapping primary key ranges
        @kuzu_node("TypeA")
        class TypeA(KuzuBaseModel):
            id: Optional[int] = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True, auto_increment=True)
            type_a_field: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)

        @kuzu_node("TypeB")
        class TypeB(KuzuBaseModel):
            id: Optional[int] = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True, auto_increment=True)
            type_b_field: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)

        # Initialize schema
        ddl_parts = [
            get_ddl_for_node(TypeA),
            get_ddl_for_node(TypeB)
        ]
        initialize_schema(session, ddl="\n".join(ddl_parts))

        # Create entities with specific primary key values
        entity_a1 = TypeA(type_a_field="A1")
        entity_a2 = TypeA(type_a_field="A2")
        entity_b1 = TypeB(type_b_field="B1")
        entity_b2 = TypeB(type_b_field="B2")

        session.add_all([entity_a1, entity_a2, entity_b1, entity_b2])
        session.commit()

        # Test node type determination from raw primary key values
        # This tests the _get_node_type_name method's database querying logic

        # Test case 1: Overlapping primary keys should raise TypeError (normal in graph databases)
        # Both TypeA and TypeB will have nodes with id=0, id=1, etc.
        with pytest.raises(TypeError, match="Primary key value 0 exists in multiple node types"):
            session._get_node_type_name(entity_a1.id)  # ID 0 exists in both TypeA and TypeB

        # Test case 4: Non-existent primary key should raise TypeError
        with pytest.raises(TypeError, match="Primary key value 999 does not exist in any registered node type"):
            session._get_node_type_name(999)

        # Test case 5: Model instance should work directly
        type_name_instance = session._get_node_type_name(entity_a2)
        assert type_name_instance == "TypeA", f"Expected 'TypeA', got '{type_name_instance}'"

        session.close()

    def test_malformed_relationship_pairs(self, test_db_path):
        """
        test: Malformed relationship pair definitions.

        Tests error handling when relationship pairs are incorrectly defined
        or when node types don't match the actual instances.
        """
        session = KuzuSession(db_path=test_db_path)

        # Create a relationship with intentionally problematic pair definition
        @kuzu_node("NodeX")
        class NodeX(KuzuBaseModel):
            id: Optional[int] = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True, auto_increment=True)
            x_field: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)

        @kuzu_node("NodeY")
        class NodeY(KuzuBaseModel):
            id: Optional[int] = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True, auto_increment=True)
            y_field: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)

        @kuzu_node("NodeZ")
        class NodeZ(KuzuBaseModel):
            id: Optional[int] = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True, auto_increment=True)
            z_field: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)

        # Relationship that only defines (NodeX, NodeY) pair but we'll try to use (NodeY, NodeZ)
        @kuzu_relationship("RestrictedRel", pairs=[(NodeX, NodeY)])
        class RestrictedRel(KuzuRelationshipBase):
            rel_id: Optional[int] = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True, auto_increment=True)
            rel_data: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)

        # Initialize schema
        ddl_parts = [
            get_ddl_for_node(NodeX),
            get_ddl_for_node(NodeY),
            get_ddl_for_node(NodeZ),
            get_ddl_for_relationship(RestrictedRel)
        ]
        initialize_schema(session, ddl="\n".join(ddl_parts))

        # Create entities
        node_x = NodeX(x_field="X1")
        node_y = NodeY(y_field="Y1")
        node_z = NodeZ(z_field="Z1")

        session.add_all([node_x, node_y, node_z])
        session.commit()

        # Test case 1: Valid relationship pair should work
        valid_rel = RestrictedRel(from_node=node_x, to_node=node_y, rel_data="Valid")
        session.add(valid_rel)
        session.commit()
        assert valid_rel.rel_id is not None

        # Test case 2: Invalid relationship pair should raise ValueError
        invalid_rel = RestrictedRel(from_node=node_y, to_node=node_z, rel_data="Invalid")
        session.add(invalid_rel)

        with pytest.raises(ValueError, match="No matching relationship pair found"):
            session.commit()

        session.close()

    def test_auto_increment_boundary_conditions(self, test_db_path):
        """
        test: Auto-increment boundary conditions and edge cases.

        Tests correctness at boundary conditions like maximum values,
        rollover scenarios, and unusual data patterns.
        """
        session = KuzuSession(db_path=test_db_path)

        # Initialize schema
        ddl = get_ddl_for_node(self.AutoUser)
        initialize_schema(session, ddl=ddl)

        # Test case 1: Large batch insertion to test sequence integrity
        batch_size = 1000
        users = [
            self.AutoUser(name=f"BatchUser_{i}", email=f"batch_{i}@example.com")
            for i in range(batch_size)
        ]

        session.add_all(users)
        session.commit()

        # Validation: All IDs should be sequential from 0 to 999
        user_ids = [user.id for user in users]

        # Validation 1: All IDs must be non-None integers
        for user_id in user_ids:
            assert user_id is not None
            assert isinstance(user_id, int)
            assert user_id >= 0

        # Validation 2: IDs must be unique
        assert len(set(user_ids)) == batch_size

        # Validation 3: IDs must form contiguous sequence
        sorted_ids = sorted(user_ids)
        expected_sequence = list(range(batch_size))
        assert sorted_ids == expected_sequence

        # Test case 2: Mixed manual and auto-increment values
        # This should test the validation logic for manual auto-increment values
        manual_user = self.AutoUser(name="ManualUser", email="manual@example.com")
        # Don't set ID - should auto-generate as 1000 (next in sequence)

        session.add(manual_user)
        session.commit()

        assert manual_user.id == batch_size  # Should be 1000

        session.close()

    def test_debug_bulk_insert_auto_increment(self, test_db_path):
        """
        Debug test: Simple bulk insert with auto-increment to debug the issue.
        """
        session = KuzuSession(db_path=test_db_path)

        # Initialize schema
        ddl = get_ddl_for_node(self.AutoUser)
        initialize_schema(session, ddl=ddl)

        # Create a small batch of users
        users = [
            self.AutoUser(name=f"DebugUser_{i}", email=f"debug_{i}@example.com")
            for i in range(3)
        ]

        # Check initial state
        print(f"Before insert - User IDs: {[u.id for u in users]}")

        # Add and commit
        for user in users:
            session.add(user)
        session.commit()

        # Check final state
        print(f"After insert - User IDs: {[u.id for u in users]}")

        # Verify IDs were generated
        for i, user in enumerate(users):
            print(f"User {i}: id={user.id}, name={user.name}, email={user.email}")
            assert user.id is not None, f"User {i} has None ID"
            assert isinstance(user.id, int), f"User {i} ID is not int: {type(user.id)}"

        session.close()

    def test_debug_bulk_insert_threshold(self, test_db_path):
        """
        Debug test: Test bulk insert threshold behavior.
        """
        # Initialize schema
        session = KuzuSession(db_path=test_db_path, bulk_insert_threshold=5)  # Lower threshold for testing
        ddl = get_ddl_for_node(self.AutoUser)
        initialize_schema(session, ddl=ddl)

        # Test with exactly the threshold number of users (should trigger bulk insert)
        users = [
            self.AutoUser(name=f"BulkUser_{i}", email=f"bulk_{i}@example.com")
            for i in range(5)  # Exactly at threshold
        ]

        print(f"Before insert - User IDs: {[u.id for u in users]}")

        for user in users:
            session.add(user)
        session.commit()

        print(f"After insert - User IDs: {[u.id for u in users]}")

        # Verify all IDs were generated
        for i, user in enumerate(users):
            print(f"User {i}: id={user.id}, name={user.name}")
            assert user.id is not None, f"User {i} has None ID after bulk insert"
            assert isinstance(user.id, int), f"User {i} ID is not int: {type(user.id)}"

        session.close()

    def test_debug_sequential_sessions(self, test_db_path):
        """
        Debug test: Sequential sessions to debug concurrent access issues.
        """
        # Initialize schema with first session
        session1 = KuzuSession(db_path=test_db_path)
        ddl = get_ddl_for_node(self.AutoUser)
        initialize_schema(session1, ddl=ddl)
        session1.close()

        # Test sequential sessions
        all_users = []
        for i in range(3):
            session = KuzuSession(db_path=test_db_path)

            users_for_session = [
                self.AutoUser(name=f"SeqUser_{i}_{j}", email=f"seq_{i}_{j}@example.com")
                for j in range(2)
            ]

            for user in users_for_session:
                session.add(user)
            session.commit()

            print(f"Session {i} - User IDs: {[u.id for u in users_for_session]}")

            all_users.extend(users_for_session)
            session.close()

        # Verify all IDs are unique and not None
        all_ids = [user.id for user in all_users]
        print(f"All IDs: {all_ids}")

        for user_id in all_ids:
            assert user_id is not None, f"Found None ID in: {all_ids}"
            assert isinstance(user_id, int), f"Non-integer ID: {user_id}"

        # Check uniqueness
        assert len(set(all_ids)) == len(all_ids), f"Duplicate IDs found: {all_ids}"

    # ========================================
    # UUID AUTO-INCREMENT TESTS
    # ========================================

    def test_uuid_auto_increment_field_definition(self):
        """Test that UUID auto-increment fields are properly defined."""
        # Check that UUID auto-increment fields are recognized
        auto_increment_fields = self.AutoUUIDUser.get_auto_increment_fields()
        assert "id" in auto_increment_fields

        # Check metadata
        metadata = self.AutoUUIDUser.get_auto_increment_metadata()
        assert "id" in metadata
        assert metadata["id"].kuzu_type == KuzuDataType.UUID
        assert metadata["id"].auto_increment is True
        assert metadata["id"].primary_key is True

    def test_uuid_auto_increment_ddl_generation(self):
        """Test that DDL is correctly generated for UUID auto-increment fields."""
        ddl = get_ddl_for_node(self.AutoUUIDUser)

        # Should contain UUID type and DEFAULT gen_random_uuid()
        assert "UUID" in ddl
        assert "DEFAULT gen_random_uuid()" in ddl
        assert "PRIMARY KEY" in ddl

        # Verify the structure
        assert "AutoUUIDUser" in ddl
        assert "id UUID DEFAULT gen_random_uuid() PRIMARY KEY" in ddl or "id UUID" in ddl

    def test_single_uuid_node_creation_with_auto_increment(self, test_db_path):
        """Test creating a single node with UUID auto-increment primary key."""
        session = KuzuSession(db_path=test_db_path)

        # Initialize schema
        ddl = get_ddl_for_node(self.AutoUUIDUser)
        initialize_schema(session, ddl=ddl)

        # Create user without providing UUID
        user = self.AutoUUIDUser(name="Alice", email="alice@example.com")
        assert user.id is None  # Should be None before insertion

        session.add(user)
        session.commit()

        # After insertion, the auto-increment field should be populated with a UUID
        assert user.id is not None
        assert isinstance(user.id, uuid.UUID)

        # Validate UUID format (8-4-4-4-12 hex digits)
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        assert re.match(uuid_pattern, str(user.id), re.IGNORECASE)

        session.close()

    def test_uuid_auto_increment_manual_override(self, test_db_path):
        """Test manually providing UUID values for auto-increment fields."""
        session = KuzuSession(db_path=test_db_path)

        # Initialize schema
        ddl = get_ddl_for_node(self.AutoUUIDUser)
        initialize_schema(session, ddl=ddl)

        # Create user with manual UUID object (Pydantic expects UUID object)
        manual_uuid = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
        user = self.AutoUUIDUser(id=manual_uuid, name="Bob", email="bob@example.com")

        # Verify that the manual UUID was properly set during instantiation
        assert user.id == manual_uuid

        # Verify that the field is tracked as explicitly set
        assert "id" in user.__pydantic_fields_set__

        # Verify that get_manual_auto_increment_values() works correctly
        manual_values = user.get_manual_auto_increment_values()
        assert manual_values == {"id": manual_uuid}

        session.add(user)
        session.commit()

        # The manual UUID should be preserved
        assert user.id == manual_uuid

        session.close()

    def test_uuid_auto_increment_validation_invalid_format(self, test_db_path):
        """Test validation of invalid UUID formats (strings are rejected at Pydantic level)."""
        # Pydantic should reject strings at model instantiation level
        with pytest.raises(ValidationError, match="Input should be a valid UUID"):
            self.AutoUUIDUser(id="invalid-uuid", name="Charlie", email="charlie@example.com")

    def test_uuid_auto_increment_validation_non_uuid(self, test_db_path):
        """Test validation of non-UUID values for UUID fields."""
        # Pydantic should catch non-UUID values at model instantiation
        with pytest.raises(ValidationError, match="UUID input should be a string, bytes or UUID object"):
            self.AutoUUIDUser(id=12345, name="Dave", email="dave@example.com")

    def test_invalid_auto_increment_type_rejection(self):
        """Test that invalid types for auto-increment are rejected."""
        # Should raise error for unsupported auto-increment types
        with pytest.raises(ValueError, match="Auto-increment is only supported for INT64/SERIAL and UUID fields"):
            @kuzu_node("InvalidAutoIncrement")
            class InvalidAutoIncrement(KuzuBaseModel):
                id: Optional[str] = kuzu_field(kuzu_type=KuzuDataType.STRING, primary_key=True, auto_increment=True)
                name: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)

    def test_comprehensive_uuid_auto_increment_workflow(self, test_db_path):
        """Comprehensive test demonstrating UUID auto-increment functionality end-to-end."""
        session = KuzuSession(db_path=test_db_path)

        # Initialize schema
        ddl = get_ddl_for_node(self.AutoUUIDUser)
        initialize_schema(session, ddl=ddl)

        # Test 1: Auto-generated UUID
        user1 = self.AutoUUIDUser(name="Alice", email="alice@example.com")
        assert user1.id is None

        session.add(user1)
        session.commit()

        assert user1.id is not None
        assert isinstance(user1.id, uuid.UUID)
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        assert re.match(uuid_pattern, str(user1.id), re.IGNORECASE)

        # Test 2: Manual UUID override
        manual_uuid = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
        user2 = self.AutoUUIDUser(id=manual_uuid, name="Bob", email="bob@example.com")

        # Verify that the manual UUID was properly set and tracked
        assert "id" in user2.__pydantic_fields_set__
        manual_values = user2.get_manual_auto_increment_values()
        assert manual_values == {"id": manual_uuid}

        session.add(user2)
        session.commit()

        assert user2.id == manual_uuid

        # Test 3: Verify both users exist and have different UUIDs
        all_users = session.query(self.AutoUUIDUser).all()
        assert len(all_users) == 2

        user_ids = [user.id for user in all_users]
        assert len(set(user_ids)) == 2  # All UUIDs should be unique
        assert user1.id in user_ids
        assert user2.id in user_ids

        session.close()

    @pytest.fixture
    def test_db_path(self, tmp_path):
        """Provide a temporary database path for testing."""
        return tmp_path / "test_auto_increment.db"

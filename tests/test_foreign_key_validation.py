# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive tests for foreign key validation.
Tests the validation of foreign key references in Kuzu models.
"""
import pytest
from typing import Optional
from pydantic import BaseModel

from kuzualchemy.kuzu_orm import (
    KuzuNodeBase,
    kuzu_node,
    kuzu_relationship,
    kuzu_field,
    KuzuDataType,
    ForeignKeyReference,
    clear_registry
)


class TestForeignKeyValidation:
    """Test suite for foreign key validation."""
    
    def setup_method(self):
        """Clear registry before each test."""
        clear_registry()
    
    def teardown_method(self):
        """Clear registry after each test."""
        clear_registry()
    
    def test_valid_foreign_key_reference(self):
        """Test validation of valid foreign key references."""
        @kuzu_node("User")
        class User(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
        
        @kuzu_node("Post")
        class Post(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            title: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
            user_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(
                    target_model=User,
                    target_field="id"
                )
            )
        
    
    def test_invalid_target_field(self):
        """Test validation catches invalid target field when foreign key is validated."""

        @kuzu_node("Author")
        class Author(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)

        @kuzu_node("Book")
        class Book(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            title: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
            author_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(
                    target_model=Author,
                    target_field="nonexistent_field"  # Invalid field
                )
            )

        # Validation happens when foreign keys are validated
        errors = Book.validate_foreign_keys()
        assert len(errors) > 0
        assert "nonexistent_field" in str(errors)
            
    
    def test_string_reference_skipped(self):
        """Test that string references are skipped during validation."""
        @kuzu_node("Order")
        class Order(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            customer_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(
                    target_model="Customer",  # String reference
                    target_field="id"
                )
            )
        
    def test_non_kuzu_model_detected(self):
        """Test that non-Kuzu models are detected during validation."""
        # Regular Pydantic model without @kuzu_node
        class RegularModel(BaseModel):
            id: int
            name: str

        @kuzu_node("ReferringModel")
        class ReferringModel(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            ref_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(
                    target_model=RegularModel,
                    target_field="id"
                )
            )

        # Validation happens when foreign keys are validated
        errors = ReferringModel.validate_foreign_keys()
        assert len(errors) > 0
        assert "not a Kuzu model" in str(errors) or "missing __kuzu_node_name__" in str(errors)
            
    def test_multiple_foreign_keys_in_same_model(self):
        """Test model with multiple foreign key references."""
        @kuzu_node("Country")
        class Country(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
        
        @kuzu_node("City")
        class City(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
            country_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(
                    target_model=Country,
                    target_field="id"
                )
            )
        
        @kuzu_node("Person")
        class Person(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
            birth_city_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(
                    target_model=City,
                    target_field="id"
                )
            )
            residence_city_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(
                    target_model=City,
                    target_field="id"
                )
            )
            nationality_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(
                    target_model=Country,
                    target_field="id"
                )
            )

    
    def test_self_referential_foreign_key(self):
        """Test self-referential foreign key."""
        @kuzu_node("Employee")
        class Employee(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
            manager_id: Optional[int] = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(
                    target_model="Employee",  # Self-reference as string
                    target_field="id"
                )
            )
        
    
    def test_cross_model_foreign_keys(self):
        """Test foreign keys between multiple models."""
        @kuzu_node("Department")
        class Department(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
        
        @kuzu_node("Staff")
        class Staff(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
            dept_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(
                    target_model=Department,
                    target_field="id"
                )
            )
        
        @kuzu_node("Project")
        class Project(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
            lead_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(
                    target_model=Staff,
                    target_field="id"
                )
            )
            dept_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(
                    target_model=Department,
                    target_field="id"
                )
            )

    
    def test_relationship_with_foreign_keys(self):
        """Test that relationships can also have foreign keys."""
        @kuzu_node("NodeA")
        class NodeA(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        @kuzu_node("NodeB")
        class NodeB(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        @kuzu_node("NodeC")
        class NodeC(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            value: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
        
        @kuzu_relationship("CONNECTS", pairs=[(NodeA, NodeB)])
        class ConnectsRel(KuzuNodeBase):
            weight: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE)
            ref_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(
                    target_model=NodeC,
                    target_field="id"
                )
            )
        
    
    def test_non_pydantic_model_reference(self):
        """Test reference to non-Pydantic class."""
        class NotAPydanticModel:
            """Regular Python class, not a Pydantic model."""
            id = 1
            name = "test"

        @kuzu_node("TestNode")
        class TestNode(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            ref_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(
                    target_model=NotAPydanticModel,
                    target_field="id"
                )
            )

        # Validation happens when foreign keys are validated
        errors = TestNode.validate_foreign_keys()
        assert len(errors) > 0
        assert ("not a valid Pydantic model" in str(errors) or
                "missing required" in str(errors) or
                "not a Kuzu model" in str(errors))
    def test_get_foreign_key_fields_method(self):
        """Test the get_foreign_key_fields method."""
        @kuzu_node("Parent")
        class Parent(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        @kuzu_node("Child")
        class Child(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            parent_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(
                    target_model=Parent,
                    target_field="id"
                )
            )
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
        
        fks = Child.get_foreign_key_fields()
        assert len(fks) == 1
        assert "parent_id" in fks
        assert fks["parent_id"].target_model == Parent
        assert fks["parent_id"].target_field == "id"
    
    def test_mixed_valid_invalid_foreign_keys(self):
        """Test model with mix of valid and invalid foreign keys."""
        @kuzu_node("ValidTarget")
        class ValidTarget(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            code: str = kuzu_field(kuzu_type=KuzuDataType.STRING)

        @kuzu_node("MixedRefs")
        class MixedRefs(KuzuNodeBase):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            # Valid reference
            valid_ref: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(
                    target_model=ValidTarget,
                    target_field="id"
                )
            )
            # Invalid reference - wrong field
            invalid_ref: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyReference(
                    target_model=ValidTarget,
                    target_field="wrong_field"
                )
            )

        # Validation happens when foreign keys are validated
        errors = MixedRefs.validate_foreign_keys()
        assert len(errors) > 0
        assert "wrong_field" in str(errors)

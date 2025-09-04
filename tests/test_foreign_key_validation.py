"""
Comprehensive tests for foreign key validation.
Tests the validation of foreign key references in Kuzu models.
"""
import pytest
from typing import Optional
from pydantic import BaseModel

from kuzualchemy.kuzu_orm import (
    kuzu_node,
    kuzu_relationship,
    KuzuBaseModel,
    kuzu_field,
    KuzuDataType,
    ForeignKeyMetadata,
    clear_registry,
    validate_all_models,
    get_registered_nodes,
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
        class User(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
        
        @kuzu_node("Post")
        class Post(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            title: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
            user_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyMetadata(
                    target_model=User,
                    target_field="id"
                )
            )
        
        # Should validate without errors
        validate_all_models()
    
    def test_invalid_target_field(self):
        """Test validation catches invalid target field."""
        @kuzu_node("Author")
        class Author(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
        
        @kuzu_node("Book")
        class Book(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            title: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
            author_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyMetadata(
                    target_model=Author,
                    target_field="nonexistent_field"  # Invalid field
                )
            )
        
        with pytest.raises(ValueError):
            validate_all_models()
    
    def test_string_reference_skipped(self):
        """Test that string references are skipped during validation."""
        @kuzu_node("Order")
        class Order(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            customer_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyMetadata(
                    target_model="Customer",  # String reference
                    target_field="id"
                )
            )
        
        # Should not error on string reference
        validate_all_models()
    
    def test_non_kuzu_model_detected(self):
        """Test that non-Kuzu models are detected during validation."""
        # Regular Pydantic model without @kuzu_node
        class RegularModel(BaseModel):
            id: int
            name: str
        
        @kuzu_node("ReferringModel")
        class ReferringModel(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            ref_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyMetadata(
                    target_model=RegularModel,
                    target_field="id"
                )
            )
        
        with pytest.raises(ValueError):
            validate_all_models()
    
    def test_multiple_foreign_keys_in_same_model(self):
        """Test model with multiple foreign key references."""
        @kuzu_node("Country")
        class Country(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
        
        @kuzu_node("City")
        class City(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
            country_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyMetadata(
                    target_model=Country,
                    target_field="id"
                )
            )
        
        @kuzu_node("Person")
        class Person(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
            birth_city_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyMetadata(
                    target_model=City,
                    target_field="id"
                )
            )
            residence_city_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyMetadata(
                    target_model=City,
                    target_field="id"
                )
            )
            nationality_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyMetadata(
                    target_model=Country,
                    target_field="id"
                )
            )
        
        validate_all_models()
    
    def test_self_referential_foreign_key(self):
        """Test self-referential foreign key."""
        @kuzu_node("Employee")
        class Employee(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
            manager_id: Optional[int] = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyMetadata(
                    target_model="Employee",  # Self-reference as string
                    target_field="id"
                )
            )
        
        validate_all_models()
    
    def test_cross_model_foreign_keys(self):
        """Test foreign keys between multiple models."""
        @kuzu_node("Department")
        class Department(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
        
        @kuzu_node("Staff")
        class Staff(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
            dept_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyMetadata(
                    target_model=Department,
                    target_field="id"
                )
            )
        
        @kuzu_node("Project")
        class Project(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
            lead_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyMetadata(
                    target_model=Staff,
                    target_field="id"
                )
            )
            dept_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyMetadata(
                    target_model=Department,
                    target_field="id"
                )
            )
        
        validate_all_models()
    
    def test_relationship_with_foreign_keys(self):
        """Test that relationships can also have foreign keys."""
        @kuzu_node("NodeA")
        class NodeA(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        @kuzu_node("NodeB")
        class NodeB(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        @kuzu_node("NodeC")
        class NodeC(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            value: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
        
        @kuzu_relationship("CONNECTS", pairs=[(NodeA, NodeB)])
        class ConnectsRel(KuzuBaseModel):
            weight: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE)
            ref_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyMetadata(
                    target_model=NodeC,
                    target_field="id"
                )
            )
        
        validate_all_models()
    
    def test_non_pydantic_model_reference(self):
        """Test reference to non-Pydantic class."""
        class NotAPydanticModel:
            """Regular Python class, not a Pydantic model."""
            id = 1
            name = "test"
        
        @kuzu_node("TestNode")
        class TestNode(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            ref_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyMetadata(
                    target_model=NotAPydanticModel,
                    target_field="id"
                )
            )
        
        with pytest.raises(ValueError):
            validate_all_models()
    
    def test_get_foreign_key_fields_method(self):
        """Test the get_foreign_key_fields method."""
        @kuzu_node("Parent")
        class Parent(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
        
        @kuzu_node("Child")
        class Child(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            parent_id: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyMetadata(
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
        class ValidTarget(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            code: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
        
        @kuzu_node("MixedRefs")
        class MixedRefs(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            # Valid reference
            valid_ref: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyMetadata(
                    target_model=ValidTarget,
                    target_field="id"
                )
            )
            # Invalid reference - wrong field
            invalid_ref: int = kuzu_field(
                kuzu_type=KuzuDataType.INT64,
                foreign_key=ForeignKeyMetadata(
                    target_model=ValidTarget,
                    target_field="wrong_field"
                )
            )
        
        with pytest.raises(ValueError):
            validate_all_models()

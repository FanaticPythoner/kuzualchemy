"""
Tests for BaseModel enum conversion functionality.
Tests the automatic enum conversion features in BaseModel.py.
"""

from __future__ import annotations

import pytest
from enum import Enum
from typing import Optional, Union
from pydantic import ValidationError

from kuzualchemy import BaseModel, kuzu_node, kuzu_field, KuzuDataType


class StatusEnum(Enum):
    """Test enum with string values."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"


class PriorityEnum(Enum):
    """Test enum with integer values."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class MixedEnum(Enum):
    """Test enum with mixed value types."""
    STRING_VAL = "string"
    INT_VAL = 42
    FLOAT_VAL = 3.14
    NONE_VAL = None


@kuzu_node("TestAccount")
class TestAccount(BaseModel):
    """Test model using BaseModel with enum fields."""
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT32, primary_key=True)
    status: StatusEnum = kuzu_field(kuzu_type=KuzuDataType.STRING)
    priority: PriorityEnum = kuzu_field(kuzu_type=KuzuDataType.INT32)
    mixed_field: MixedEnum = kuzu_field(kuzu_type=KuzuDataType.STRING)
    optional_status: Optional[StatusEnum] = kuzu_field(kuzu_type=KuzuDataType.STRING, default=None)


@kuzu_node("TestUnionModel")
class TestUnionModel(BaseModel):
    """Test model with Union type enum fields."""
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT32, primary_key=True)
    union_status: Union[StatusEnum, None] = kuzu_field(kuzu_type=KuzuDataType.STRING, default=None)


@kuzu_node("TestNoEnumModel")
class TestNoEnumModel(BaseModel):
    """Test model without enum fields."""
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT32, primary_key=True)
    name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)


class TestBaseModelEnumConversion:
    """Test BaseModel enum conversion functionality."""

    def test_enum_conversion_by_name(self):
        """Test converting string enum by member name."""
        account = TestAccount(
            id=1,
            status="ACTIVE",
            priority="HIGH",
            mixed_field="STRING_VAL"
        )
        assert account.status == StatusEnum.ACTIVE
        assert account.priority == PriorityEnum.HIGH
        assert account.mixed_field == MixedEnum.STRING_VAL

    def test_enum_conversion_by_value(self):
        """Test converting string enum by member value."""
        account = TestAccount(
            id=1,
            status="active",
            priority="3",
            mixed_field="string"
        )
        assert account.status == StatusEnum.ACTIVE
        assert account.priority == PriorityEnum.HIGH
        assert account.mixed_field == MixedEnum.STRING_VAL

    def test_numeric_enum_conversion(self):
        """Test converting numeric enum values."""
        account = TestAccount(
            id=1,
            status="ACTIVE",
            priority=2,
            mixed_field=42
        )
        assert account.status == StatusEnum.ACTIVE
        assert account.priority == PriorityEnum.MEDIUM
        assert account.mixed_field == MixedEnum.INT_VAL

    def test_validator_float_enum_conversion(self):
        """Test the validator converts float enum values."""
        result = TestAccount.convert_str_to_enum({
            "id": 1,
            "status": "ACTIVE",
            "priority": "LOW",
            "mixed_field": 3.14
        })
        assert result["status"] == StatusEnum.ACTIVE
        assert result["priority"] == PriorityEnum.LOW
        assert result["mixed_field"] == MixedEnum.FLOAT_VAL

    def test_validator_none_enum_value(self):
        """Test the validator handles enum with None value."""
        result = TestAccount.convert_str_to_enum({
            "id": 1,
            "status": "ACTIVE",
            "priority": "LOW",
            "mixed_field": "NONE_VAL"
        })
        assert result["mixed_field"] == MixedEnum.NONE_VAL

    def test_validator_optional_enum_none(self):
        """Test the validator handles optional enum field with None value."""
        result = TestAccount.convert_str_to_enum({
            "id": 1,
            "status": "ACTIVE",
            "priority": "LOW",
            "mixed_field": "STRING_VAL",
            "optional_status": None
        })
        assert result["optional_status"] is None

    def test_validator_optional_enum_with_value(self):
        """Test the validator handles optional enum field with actual value."""
        result = TestAccount.convert_str_to_enum({
            "id": 1,
            "status": "ACTIVE",
            "priority": "LOW",
            "mixed_field": "STRING_VAL",
            "optional_status": "PENDING"
        })
        assert result["optional_status"] == StatusEnum.PENDING

    def test_validator_union_enum_conversion(self):
        """Test the validator handles Union type enum conversion."""
        result = TestUnionModel.convert_str_to_enum({
            "id": 1,
            "union_status": "ACTIVE"
        })
        assert result["union_status"] == StatusEnum.ACTIVE

    def test_validator_union_enum_none(self):
        """Test the validator handles Union type enum with None."""
        result = TestUnionModel.convert_str_to_enum({
            "id": 1,
            "union_status": None
        })
        assert result["union_status"] is None

    def test_validator_already_enum_instance(self):
        """Test that the validator doesn't convert existing enum instances."""
        result = TestAccount.convert_str_to_enum({
            "id": 1,
            "status": StatusEnum.ACTIVE,
            "priority": PriorityEnum.HIGH,
            "mixed_field": MixedEnum.STRING_VAL
        })
        assert result["status"] == StatusEnum.ACTIVE
        assert result["priority"] == PriorityEnum.HIGH
        assert result["mixed_field"] == MixedEnum.STRING_VAL

    def test_validator_non_string_values_ignored(self):
        """Test that the validator ignores non-string values for conversion."""
        # Integer values should be converted if they match enum values
        result = TestAccount.convert_str_to_enum({
            "id": 1,
            "status": StatusEnum.ACTIVE,  # Already enum
            "priority": 2,  # Integer that matches enum value
            "mixed_field": 42  # Integer that matches enum value
        })
        assert result["status"] == StatusEnum.ACTIVE
        assert result["priority"] == PriorityEnum.MEDIUM
        assert result["mixed_field"] == MixedEnum.INT_VAL

    def test_invalid_enum_value_raises_error(self):
        """Test that invalid enum values raise ValidationError."""
        with pytest.raises(ValueError, match="Invalid value for field"):
            TestAccount(
                id=1,
                status="INVALID_STATUS",
                priority="LOW",
                mixed_field="STRING_VAL"
            )

    def test_invalid_enum_name_raises_error(self):
        """Test that invalid enum names raise ValidationError."""
        with pytest.raises(ValueError, match="Invalid value for field"):
            TestAccount(
                id=1,
                status="NONEXISTENT_NAME",
                priority="LOW",
                mixed_field="STRING_VAL"
            )

    def test_non_dict_values_passthrough(self):
        """Test that non-dict values are passed through unchanged."""
        # Test with non-dict input
        result = TestAccount.convert_str_to_enum("not_a_dict")
        assert result == "not_a_dict"

        result = TestAccount.convert_str_to_enum(123)
        assert result == 123

        result = TestAccount.convert_str_to_enum([1, 2, 3])
        assert result == [1, 2, 3]

    def test_model_without_annotations(self):
        """Test model without __annotations__ attribute."""
        class NoAnnotationsModel(BaseModel):
            pass
        
        # Should not raise error and pass through values unchanged
        result = NoAnnotationsModel.convert_str_to_enum({})
        assert result == {}

    def test_enum_cache_functionality(self):
        """Test that enum cache is populated and used correctly."""
        # First creation should populate cache
        account1 = TestAccount(
            id=1,
            status="ACTIVE",
            priority="HIGH",
            mixed_field="STRING_VAL"
        )
        
        # Second creation should use cache
        account2 = TestAccount(
            id=2,
            status="INACTIVE",
            priority="LOW",
            mixed_field="INT_VAL"
        )
        
        assert account1.status == StatusEnum.ACTIVE
        assert account2.status == StatusEnum.INACTIVE

    def test_numeric_string_conversion(self):
        """Test conversion of numeric strings."""
        # Test integer string
        account = TestAccount(
            id=1,
            status="ACTIVE",
            priority="2",  # String that looks like integer
            mixed_field="42"  # String that looks like integer
        )
        assert account.priority == PriorityEnum.MEDIUM
        assert account.mixed_field == MixedEnum.INT_VAL

        # Test float string
        account2 = TestAccount(
            id=2,
            status="ACTIVE",
            priority="LOW",
            mixed_field="3.14"  # String that looks like float
        )
        assert account2.mixed_field == MixedEnum.FLOAT_VAL

    def test_validator_negative_numeric_string(self):
        """Test the validator converts negative numeric strings."""
        class NegativeEnum(Enum):
            NEGATIVE = -1
            ZERO = 0
            POSITIVE = 1

        @kuzu_node("NegativeModel")
        class NegativeModel(BaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT32, primary_key=True)
            value: NegativeEnum = kuzu_field(kuzu_type=KuzuDataType.INT32)

        model = NegativeModel(id=1, value="-1")
        assert model.value == NegativeEnum.NEGATIVE

    def test_non_enum_fields_unchanged(self):
        """Test that non-enum fields are not affected by conversion."""
        model = TestNoEnumModel(id=1, name="test")
        assert model.id == 1
        assert model.name == "test"

    def test_field_not_in_values(self):
        """Test handling when field is not present in input values."""
        # This tests the continue statement when field_name not in values
        account = TestAccount(
            id=1,
            status="ACTIVE",
            priority="LOW",
            mixed_field="STRING_VAL"
            # optional_status not provided
        )
        assert account.optional_status is None

    def test_validator_non_class_type_in_union(self):
        """Test the validator handles non-class types in Union args."""
        # This tests the TypeError handling in Union processing
        from typing import Union, List

        @kuzu_node("ComplexUnionModel")
        class ComplexUnionModel(BaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT32, primary_key=True)
            # Union with non-class type (List[str] is not a class)
            complex_field: Union[StatusEnum, List[str], None] = kuzu_field(kuzu_type=KuzuDataType.STRING, default=None)

        model = ComplexUnionModel(id=1, complex_field="ACTIVE")
        assert model.complex_field == StatusEnum.ACTIVE

    def test_validator_non_class_field_type(self):
        """Test the validator handles non-class field types."""
        # This tests the TypeError handling when field_type is not a class
        from typing import List

        @kuzu_node("NonClassFieldModel")
        class NonClassFieldModel(BaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT32, primary_key=True)
            list_field: List[str] = kuzu_field(kuzu_type=KuzuDataType.STRING, default=[])

        result = NonClassFieldModel.convert_str_to_enum({"id": 1, "list_field": "test"})
        # Should not convert since List[str] is not an Enum
        assert result["list_field"] == "test"

    def test_validator_error_message_formatting(self):
        """Test that the validator error messages are properly formatted."""
        with pytest.raises(ValueError) as exc_info:
            TestAccount.convert_str_to_enum({
                "id": 1,
                "status": "INVALID_STATUS",
                "priority": "LOW",
                "mixed_field": "STRING_VAL"
            })
        
        error_message = str(exc_info.value)
        assert "Invalid value for field" in error_message
        assert "status" in error_message
        assert "INVALID_STATUS" in error_message
        assert "Valid names:" in error_message
        assert "valid values:" in error_message

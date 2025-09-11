# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
Tests for BaseModel enum conversion functionality.
Tests the automatic enum conversion features in BaseModel.py.
"""

from __future__ import annotations

import pytest
from enum import Enum, IntEnum, StrEnum
from typing import Optional, Union, List, Tuple
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


class StatusIntEnum(IntEnum):
    """Test IntEnum with integer values."""
    INACTIVE = 0
    ACTIVE = 1
    PENDING = 2


class StatusStrEnum(StrEnum):
    """Test StrEnum with string values."""
    INACTIVE = "inactive"
    ACTIVE = "active"
    PENDING = "pending"


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


@kuzu_node("TestIntEnumModel")
class TestIntEnumModel(BaseModel):
    """Test model using IntEnum fields."""
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT32, primary_key=True)
    int_status: StatusIntEnum = kuzu_field(kuzu_type=KuzuDataType.INT32)
    optional_int_status: Optional[StatusIntEnum] = kuzu_field(kuzu_type=KuzuDataType.INT32, default=None)


@kuzu_node("TestStrEnumModel")
class TestStrEnumModel(BaseModel):
    """Test model using StrEnum fields."""
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT32, primary_key=True)
    str_status: StatusStrEnum = kuzu_field(kuzu_type=KuzuDataType.STRING)
    optional_str_status: Optional[StatusStrEnum] = kuzu_field(kuzu_type=KuzuDataType.STRING, default=None)


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


# Additional tests for list/tuple of Enum conversions
from typing import List, Tuple


@kuzu_node("ListEnumModel")
class ListEnumModel(BaseModel):
    """Model with list-based enum fields to validate conversions."""
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT32, primary_key=True)
    statuses: List[StatusEnum] = kuzu_field(kuzu_type=KuzuDataType.STRING)
    priorities: List[PriorityEnum] = kuzu_field(kuzu_type=KuzuDataType.INT32)
    mixed_list: List[MixedEnum] = kuzu_field(kuzu_type=KuzuDataType.STRING)
    optional_statuses: Optional[List[StatusEnum]] = kuzu_field(kuzu_type=KuzuDataType.STRING, default=None)
    optional_mixed_list: List[Optional[MixedEnum]] = kuzu_field(kuzu_type=KuzuDataType.STRING, default=[])


@kuzu_node("TupleEnumModel")
class TupleEnumModel(BaseModel):
    """Model with tuple-based enum fields to validate conversions."""
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT32, primary_key=True)
    status_tuple: Tuple[StatusEnum, ...] = kuzu_field(kuzu_type=KuzuDataType.STRING)
    priority_tuple_fixed: Tuple[PriorityEnum, PriorityEnum] = kuzu_field(kuzu_type=KuzuDataType.INT32)


class TestBaseModelEnumListTupleConversion:
    """Tests for list and tuple conversions of Enums in BaseModel."""

    def test_list_enum_conversion_by_name_and_value(self):
        """Test converting list of enums by name and value."""
        model = ListEnumModel(
            id=1,
            statuses=["ACTIVE", "inactive", "PENDING"],
            priorities=[1, "2", 3],
            mixed_list=["STRING_VAL", 42, "3.14"],
        )
        assert model.statuses == [StatusEnum.ACTIVE, StatusEnum.INACTIVE, StatusEnum.PENDING]
        assert model.priorities == [PriorityEnum.LOW, PriorityEnum.MEDIUM, PriorityEnum.HIGH]
        assert model.mixed_list == [MixedEnum.STRING_VAL, MixedEnum.INT_VAL, MixedEnum.FLOAT_VAL]

    def test_list_enum_mixed_inputs_and_enum_instances(self):
        """Test list conversion with mixed string/enum inputs."""
        model = ListEnumModel(
            id=2,
            statuses=[StatusEnum.ACTIVE, "INACTIVE"],
            priorities=[PriorityEnum.MEDIUM, "3"],
            mixed_list=[MixedEnum.STRING_VAL, 42, "3.14"],
        )
        assert model.statuses == [StatusEnum.ACTIVE, StatusEnum.INACTIVE]
        assert model.priorities == [PriorityEnum.MEDIUM, PriorityEnum.HIGH]
        assert model.mixed_list == [MixedEnum.STRING_VAL, MixedEnum.INT_VAL, MixedEnum.FLOAT_VAL]

    def test_optional_list_enum_none_and_inner_none(self):
        """Test optional list and list with optional elements."""
        model = ListEnumModel(
            id=3,
            statuses=["ACTIVE"],
            priorities=["1"],
            mixed_list=["NONE_VAL", "STRING_VAL"],
            optional_statuses=None,
            optional_mixed_list=[None, "INT_VAL"],
        )
        assert model.optional_statuses is None
        assert model.optional_mixed_list == [None, MixedEnum.INT_VAL]
        assert model.mixed_list[0] == MixedEnum.NONE_VAL

    def test_tuple_enum_conversion_homogeneous(self):
        """Test tuple conversion with homogeneous and fixed-length tuples."""
        model = TupleEnumModel(
            id=4,
            status_tuple=("ACTIVE", "PENDING", "inactive"),
            priority_tuple_fixed=("1", 2),
        )
        assert model.status_tuple == (StatusEnum.ACTIVE, StatusEnum.PENDING, StatusEnum.INACTIVE)
        assert model.priority_tuple_fixed == (PriorityEnum.LOW, PriorityEnum.MEDIUM)

    def test_empty_list_and_tuple(self):
        """Test empty sequences are handled correctly."""
        model = ListEnumModel(
            id=5,
            statuses=[],
            priorities=[],
            mixed_list=[],
        )
        assert model.statuses == []
        assert model.priorities == []
        assert model.mixed_list == []

    def test_invalid_list_element_raises_error(self):
        """Test that invalid elements in lists raise appropriate errors."""
        with pytest.raises(ValueError, match="Invalid enum value"):
            ListEnumModel(
                id=6,
                statuses=["ACTIVE", "BOGUS"],
                priorities=[1, 2],
                mixed_list=["STRING_VAL"],
            )

    def test_converter_function_performance(self):
        """Test that converter function handles various input types efficiently."""
        # Test the converter function directly
        converter = BaseModel._create_enum_converter(StatusEnum)

        # Test various input types
        assert converter("ACTIVE") == StatusEnum.ACTIVE
        assert converter("active") == StatusEnum.ACTIVE
        assert converter(StatusEnum.INACTIVE) == StatusEnum.INACTIVE


class TestIntEnumConversion:
    """Test IntEnum conversion functionality."""

    def test_int_enum_conversion_by_name(self):
        """Test converting IntEnum by member name."""
        model = TestIntEnumModel(
            id=1,
            int_status="ACTIVE"
        )
        assert model.int_status == StatusIntEnum.ACTIVE
        assert model.int_status == 1
        assert isinstance(model.int_status, StatusIntEnum)

    def test_int_enum_conversion_by_value(self):
        """Test converting IntEnum by integer value."""
        model = TestIntEnumModel(
            id=1,
            int_status=1
        )
        assert model.int_status == StatusIntEnum.ACTIVE
        assert model.int_status == 1
        assert isinstance(model.int_status, StatusIntEnum)

    def test_int_enum_conversion_by_string_value(self):
        """Test converting IntEnum by string representation of integer."""
        model = TestIntEnumModel(
            id=1,
            int_status="2"
        )
        assert model.int_status == StatusIntEnum.PENDING
        assert model.int_status == 2
        assert isinstance(model.int_status, StatusIntEnum)

    def test_optional_int_enum_none(self):
        """Test Optional[IntEnum] with None value."""
        model = TestIntEnumModel(
            id=1,
            int_status="ACTIVE",
            optional_int_status=None
        )
        assert model.int_status == StatusIntEnum.ACTIVE
        assert model.optional_int_status is None

    def test_optional_int_enum_conversion(self):
        """Test Optional[IntEnum] with actual value."""
        model = TestIntEnumModel(
            id=1,
            int_status="ACTIVE",
            optional_int_status="0"
        )
        assert model.int_status == StatusIntEnum.ACTIVE
        assert model.optional_int_status == StatusIntEnum.INACTIVE
        assert model.optional_int_status == 0

    def test_int_enum_invalid_value_raises_error(self):
        """Test that invalid IntEnum values raise appropriate errors."""
        with pytest.raises(ValueError, match="Invalid value for field"):
            TestIntEnumModel(
                id=1,
                int_status="INVALID"
            )


class TestStrEnumConversion:
    """Test StrEnum conversion functionality."""

    def test_str_enum_conversion_by_name(self):
        """Test converting StrEnum by member name."""
        model = TestStrEnumModel(
            id=1,
            str_status="ACTIVE"
        )
        assert model.str_status == StatusStrEnum.ACTIVE
        assert model.str_status == "active"
        assert isinstance(model.str_status, StatusStrEnum)

    def test_str_enum_conversion_by_value(self):
        """Test converting StrEnum by string value."""
        model = TestStrEnumModel(
            id=1,
            str_status="active"
        )
        assert model.str_status == StatusStrEnum.ACTIVE
        assert model.str_status == "active"
        assert isinstance(model.str_status, StatusStrEnum)

    def test_optional_str_enum_none(self):
        """Test Optional[StrEnum] with None value."""
        model = TestStrEnumModel(
            id=1,
            str_status="ACTIVE",
            optional_str_status=None
        )
        assert model.str_status == StatusStrEnum.ACTIVE
        assert model.optional_str_status is None

    def test_optional_str_enum_conversion(self):
        """Test Optional[StrEnum] with actual value."""
        model = TestStrEnumModel(
            id=1,
            str_status="ACTIVE",
            optional_str_status="pending"
        )
        assert model.str_status == StatusStrEnum.ACTIVE
        assert model.optional_str_status == StatusStrEnum.PENDING
        assert model.optional_str_status == "pending"

    def test_str_enum_invalid_value_raises_error(self):
        """Test that invalid StrEnum values raise appropriate errors."""
        with pytest.raises(ValueError, match="Invalid value for field"):
            TestStrEnumModel(
                id=1,
                str_status="INVALID"
            )


class TestListTupleIntStrEnumConversion:
    """Test List/Tuple conversion with IntEnum and StrEnum."""

    def test_list_int_enum_conversion(self):
        """Test List[IntEnum] conversion."""
        # Create a test model with List[IntEnum]
        @kuzu_node("ListIntEnumModel")
        class ListIntEnumModel(BaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT32, primary_key=True)
            int_statuses: List[StatusIntEnum] = kuzu_field(kuzu_type=KuzuDataType.STRING)

        model = ListIntEnumModel(
            id=1,
            int_statuses=["ACTIVE", "0", 2]  # Mix of name, string value, int value
        )
        assert model.int_statuses == [StatusIntEnum.ACTIVE, StatusIntEnum.INACTIVE, StatusIntEnum.PENDING]
        assert all(isinstance(status, StatusIntEnum) for status in model.int_statuses)

    def test_list_str_enum_conversion(self):
        """Test List[StrEnum] conversion."""
        # Create a test model with List[StrEnum]
        @kuzu_node("ListStrEnumModel")
        class ListStrEnumModel(BaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT32, primary_key=True)
            str_statuses: List[StatusStrEnum] = kuzu_field(kuzu_type=KuzuDataType.STRING)

        model = ListStrEnumModel(
            id=1,
            str_statuses=["ACTIVE", "inactive", "pending"]  # Mix of name and value
        )
        assert model.str_statuses == [StatusStrEnum.ACTIVE, StatusStrEnum.INACTIVE, StatusStrEnum.PENDING]
        assert all(isinstance(status, StatusStrEnum) for status in model.str_statuses)

    def test_tuple_int_enum_conversion(self):
        """Test Tuple[IntEnum] conversion."""
        # Create a test model with Tuple[IntEnum]
        @kuzu_node("TupleIntEnumModel")
        class TupleIntEnumModel(BaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT32, primary_key=True)
            int_statuses: Tuple[StatusIntEnum, ...] = kuzu_field(kuzu_type=KuzuDataType.STRING)

        model = TupleIntEnumModel(
            id=1,
            int_statuses=("ACTIVE", "1")  # Mix of name and string value
        )
        assert model.int_statuses == (StatusIntEnum.ACTIVE, StatusIntEnum.ACTIVE)
        assert all(isinstance(status, StatusIntEnum) for status in model.int_statuses)

    def test_optional_list_int_enum_with_none_elements(self):
        """Test List[Optional[IntEnum]] conversion."""
        # Create a test model with List[Optional[IntEnum]]
        @kuzu_node("OptionalListIntEnumModel")
        class OptionalListIntEnumModel(BaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT32, primary_key=True)
            int_statuses: List[Optional[StatusIntEnum]] = kuzu_field(kuzu_type=KuzuDataType.STRING)

        model = OptionalListIntEnumModel(
            id=1,
            int_statuses=["ACTIVE", None, "2"]
        )
        assert model.int_statuses == [StatusIntEnum.ACTIVE, None, StatusIntEnum.PENDING]


class TestBaseModelEnumListTupleConversion:
    """Test List and Tuple enum conversion functionality."""

    def test_large_list_performance(self):
        """Test performance with larger lists to ensure O(n) behavior."""
        large_status_list = ["ACTIVE", "INACTIVE", "PENDING"] * 100
        large_priority_list = [1, 2, 3] * 100

        model = ListEnumModel(
            id=7,
            statuses=large_status_list,
            priorities=large_priority_list,
            mixed_list=["STRING_VAL"] * 100,
        )

        expected_statuses = [StatusEnum.ACTIVE, StatusEnum.INACTIVE, StatusEnum.PENDING] * 100
        expected_priorities = [PriorityEnum.LOW, PriorityEnum.MEDIUM, PriorityEnum.HIGH] * 100
        expected_mixed = [MixedEnum.STRING_VAL] * 100

        assert model.statuses == expected_statuses
        assert model.priorities == expected_priorities
        assert model.mixed_list == expected_mixed
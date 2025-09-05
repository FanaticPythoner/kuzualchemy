# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive tests for typed array specifications.
Tests the various ways to specify typed arrays in Kuzu fields.
"""
import pytest
from typing import List

from kuzualchemy.kuzu_orm import (
    kuzu_node,
    KuzuBaseModel,
    kuzu_field,
    KuzuDataType,
    ArrayTypeSpecification,
    clear_registry,
    generate_node_ddl,
)


class TestTypedArrays:
    """Test suite for typed array specifications."""
    
    def setup_method(self):
        """Clear registry before each test."""
        clear_registry()
    
    def teardown_method(self):
        """Clear registry after each test."""
        clear_registry()
    
    def test_array_specification_with_string_syntax(self):
        """Test array specification using string syntax like 'INT64[]'."""
        @kuzu_node("ArrayNode1")
        class ArrayNode1(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            int_array: List[int] = kuzu_field(kuzu_type="INT64[]")
            string_array: List[str] = kuzu_field(kuzu_type="STRING[]")
            double_array: List[float] = kuzu_field(kuzu_type="DOUBLE[]")
            bool_array: List[bool] = kuzu_field(kuzu_type="BOOL[]")
        
        ddl = generate_node_ddl(ArrayNode1)
        
        assert "int_array INT64[]" in ddl
        assert "string_array STRING[]" in ddl
        assert "double_array DOUBLE[]" in ddl
        assert "bool_array BOOL[]" in ddl
    
    def test_array_specification_with_element_type_param(self):
        """Test array specification using element_type parameter."""
        @kuzu_node("ArrayNode2")
        class ArrayNode2(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            int_array: List[int] = kuzu_field(
                kuzu_type=KuzuDataType.ARRAY,
                element_type=KuzuDataType.INT64
            )
            string_array: List[str] = kuzu_field(
                kuzu_type=KuzuDataType.ARRAY,
                element_type="STRING"
            )
        
        ddl = generate_node_ddl(ArrayNode2)
        
        assert "int_array INT64[]" in ddl
        assert "string_array STRING[]" in ddl
    
    def test_array_specification_direct_object(self):
        """Test array specification using ArrayTypeSpecification directly."""
        @kuzu_node("ArrayNode3")
        class ArrayNode3(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            int_array: List[int] = kuzu_field(
                kuzu_type=ArrayTypeSpecification(element_type=KuzuDataType.INT64)
            )
            float_array: List[float] = kuzu_field(
                kuzu_type=ArrayTypeSpecification(element_type="DOUBLE")
            )
        
        ddl = generate_node_ddl(ArrayNode3)
        
        assert "int_array INT64[]" in ddl
        assert "float_array DOUBLE[]" in ddl
    
    def test_all_primitive_array_types(self):
        """Test all primitive data types as array elements."""
        @kuzu_node("AllArrayTypes")
        class AllArrayTypes(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            
            # Integer types
            int8_arr: List[int] = kuzu_field(kuzu_type="INT8[]")
            int16_arr: List[int] = kuzu_field(kuzu_type="INT16[]")
            int32_arr: List[int] = kuzu_field(kuzu_type="INT32[]")
            int64_arr: List[int] = kuzu_field(kuzu_type="INT64[]")
            int128_arr: List[int] = kuzu_field(kuzu_type="INT128[]")
            
            # Unsigned integer types
            uint8_arr: List[int] = kuzu_field(kuzu_type="UINT8[]")
            uint16_arr: List[int] = kuzu_field(kuzu_type="UINT16[]")
            uint32_arr: List[int] = kuzu_field(kuzu_type="UINT32[]")
            uint64_arr: List[int] = kuzu_field(kuzu_type="UINT64[]")
            
            # Float types
            float_arr: List[float] = kuzu_field(kuzu_type="FLOAT[]")
            double_arr: List[float] = kuzu_field(kuzu_type="DOUBLE[]")
            
            # String and bool
            string_arr: List[str] = kuzu_field(kuzu_type="STRING[]")
            bool_arr: List[bool] = kuzu_field(kuzu_type="BOOL[]")
        
        ddl = generate_node_ddl(AllArrayTypes)
        
        # Check all array types are present
        assert "int8_arr INT8[]" in ddl
        assert "int16_arr INT16[]" in ddl
        assert "int32_arr INT32[]" in ddl
        assert "int64_arr INT64[]" in ddl
        assert "int128_arr INT128[]" in ddl
        assert "uint8_arr UINT8[]" in ddl
        assert "uint16_arr UINT16[]" in ddl
        assert "uint32_arr UINT32[]" in ddl
        assert "uint64_arr UINT64[]" in ddl
        assert "float_arr FLOAT[]" in ddl
        assert "double_arr DOUBLE[]" in ddl
        assert "string_arr STRING[]" in ddl
        assert "bool_arr BOOL[]" in ddl
    
    def test_mixed_array_specifications_in_same_model(self):
        """Test mixing different array specification methods in the same model."""
        @kuzu_node("MixedArrayNode")
        class MixedArrayNode(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            
            # String syntax
            arr1: List[int] = kuzu_field(kuzu_type="INT64[]")
            
            # element_type parameter
            arr2: List[str] = kuzu_field(
                kuzu_type=KuzuDataType.ARRAY,
                element_type=KuzuDataType.STRING
            )
            
            # ArrayTypeSpecification object
            arr3: List[float] = kuzu_field(
                kuzu_type=ArrayTypeSpecification(element_type=KuzuDataType.DOUBLE)
            )
        
        ddl = generate_node_ddl(MixedArrayNode)
        
        assert "arr1 INT64[]" in ddl
        assert "arr2 STRING[]" in ddl
        assert "arr3 DOUBLE[]" in ddl
    
    def test_array_with_not_null_constraint(self):
        """Test arrays with NOT NULL constraints (ignored in NODE tables per Kuzu v0.11.2)."""
        @kuzu_node("ArrayConstraints")
        class ArrayConstraints(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            required_array: List[int] = kuzu_field(kuzu_type="INT64[]", not_null=True)
            optional_array: List[int] = kuzu_field(kuzu_type="INT64[]", not_null=False)
        
        ddl = generate_node_ddl(ArrayConstraints)
        
        # NOT NULL is not supported in NODE tables, so it's ignored
        assert "required_array INT64[]" in ddl
        assert "optional_array INT64[]" in ddl
        assert "NOT NULL" not in ddl  # NOT NULL not supported in NODE tables
    
    def test_array_with_default_values(self):
        """Test arrays with default values."""
        @kuzu_node("ArrayDefaults")
        class ArrayDefaults(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            int_array: List[int] = kuzu_field(kuzu_type="INT64[]", default=[1, 2, 3])
            string_array: List[str] = kuzu_field(kuzu_type="STRING[]", default=["a", "b"])
            empty_array: List[int] = kuzu_field(kuzu_type="INT64[]", default=[])
        
        ddl = generate_node_ddl(ArrayDefaults)
        
        # Check for default values
        assert "[1, 2, 3]" in ddl
        assert '["a", "b"]' in ddl or "['a', 'b']" in ddl  # Either quote style
        assert "[]" in ddl  # Empty array default
    
    def test_invalid_array_syntax_fallback(self):
        """Test that invalid array syntax doesn't crash but stores as custom type."""
        @kuzu_node("CustomArrayNode")
        class CustomArrayNode(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            # Custom type that's not in KuzuDataType enum
            custom_array: List[str] = kuzu_field(kuzu_type="CUSTOM_TYPE[]")
        
        ddl = generate_node_ddl(CustomArrayNode)
        
        # Should still generate with custom type
        assert "custom_array CUSTOM_TYPE[]" in ddl
    
    def test_element_type_overrides_array_type(self):
        """Test that element_type parameter takes precedence when specified."""
        @kuzu_node("OverrideNode")
        class OverrideNode(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            # ARRAY type but element_type specifies INT64
            arr: List[int] = kuzu_field(
                kuzu_type=KuzuDataType.ARRAY,
                element_type=KuzuDataType.INT64
            )
        
        ddl = generate_node_ddl(OverrideNode)
        
        assert "arr INT64[]" in ddl
    
    def test_case_insensitive_array_types(self):
        """Test that array type strings work regardless of case."""
        @kuzu_node("CaseNode")
        class CaseNode(KuzuBaseModel):
            id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            lower_array: List[int] = kuzu_field(kuzu_type="int64[]")
            mixed_array: List[str] = kuzu_field(kuzu_type="StRiNg[]")
        
        # Should not raise errors during model creation
        # DDL generation handles case normalization
        ddl = generate_node_ddl(CaseNode)
        assert ddl  # Just verify it generates without error
    
    def test_array_cannot_be_primary_key(self):
        """Test that arrays cannot be used as primary keys."""
        with pytest.raises(ValueError, match="Arrays cannot be used as primary keys"):
            @kuzu_node("InvalidPK")
            class InvalidPK(KuzuBaseModel):
                id: List[int] = kuzu_field(
                    kuzu_type="INT64[]",
                    primary_key=True  # This should fail
                )

# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for List/String Operators Implementation.
"""

import pytest

from kuzualchemy.kuzu_query_fields import QueryField
from kuzualchemy.kuzu_query_expressions import (
    ArithmeticExpression, ArithmeticOperator, FunctionExpression,
    FieldFilterExpression, ComparisonOperator
)


class TestListStringOperatorsImplementation:
    """Tests for all 5 list/string operators."""
    
    # ============================================================================
    # LIST CONCATENATION OPERATOR TESTS (+)
    # ============================================================================
    
    def test_list_concatenation_basic(self):
        """Test list concatenation operator (+) - Basic functionality."""
        list_field = QueryField("tags")
        other_list = [4, 5]
        expr = list_field + other_list
        
        # Validate expression type and structure
        assert isinstance(expr, ArithmeticExpression)
        assert expr.operator == ArithmeticOperator.ADD
        assert expr.left == list_field
        assert expr.right == other_list
        
        # Validate Cypher generation
        cypher = expr.to_cypher({"n": "n"})
        params = expr.get_parameters()
        # ArithmeticExpression uses arith_right_ prefix for right operand parameters
        param_name = [k for k in params.keys() if k.startswith('arith_right_')][0]
        assert cypher == f"(n.tags + ${param_name})"
    
    def test_list_concatenation_with_field(self):
        """Test list concatenation with another field."""
        list_field1 = QueryField("first_list")
        list_field2 = QueryField("second_list")
        expr = list_field1 + list_field2
        
        assert isinstance(expr, ArithmeticExpression)
        assert expr.operator == ArithmeticOperator.ADD
        assert expr.left == list_field1
        assert expr.right == list_field2
        
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "(n.first_list + n.second_list)"
    
    def test_list_concatenation_right_hand(self):
        """Test right-hand list concatenation ([1,2,3] + field)."""
        list_field = QueryField("additional_items")
        base_list = [1, 2, 3]
        expr = base_list + list_field
        
        assert isinstance(expr, ArithmeticExpression)
        assert expr.operator == ArithmeticOperator.ADD
        assert expr.left == base_list
        assert expr.right == list_field
        
        cypher = expr.to_cypher({"n": "n"})
        params = expr.get_parameters()
        # ArithmeticExpression uses arith_left_ prefix for left operand parameters
        param_name = [k for k in params.keys() if k.startswith('arith_left_')][0]
        assert cypher == f"(${param_name} + n.additional_items)"
    
    def test_string_concatenation(self):
        """Test string concatenation using + operator."""
        string_field = QueryField("first_name")
        suffix = " Jr."
        expr = string_field + suffix
        
        assert isinstance(expr, ArithmeticExpression)
        assert expr.operator == ArithmeticOperator.ADD
        assert expr.left == string_field
        assert expr.right == suffix
        
        cypher = expr.to_cypher({"n": "n"})
        params = expr.get_parameters()
        # ArithmeticExpression uses arith_right_ prefix for right operand parameters
        param_name = [k for k in params.keys() if k.startswith('arith_right_')][0]
        assert cypher == f"(n.first_name + ${param_name})"
    
    # ============================================================================
    # LIST/STRING SLICING OPERATOR TESTS ([start:end])
    # ============================================================================
    
    def test_list_slicing_basic(self):
        """Test list slicing operator [start:end] - Basic functionality."""
        list_field = QueryField("items")
        expr = list_field[1:3]
        
        # Validate expression type and structure
        assert isinstance(expr, FunctionExpression)
        assert expr.function_name == "array_slice"
        assert expr.args[0] == list_field
        assert expr.args[1] == 1  # start
        assert expr.args[2] == 3  # stop
        
        # Validate Cypher generation
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "array_slice(n.items, 1, 3)"
    
    def test_list_slicing_with_none_start(self):
        """Test list slicing with None start ([:3])."""
        list_field = QueryField("data")
        expr = list_field[:5]
        
        assert isinstance(expr, FunctionExpression)
        assert expr.function_name == "array_slice"
        assert expr.args[0] == list_field
        assert expr.args[1] == 1  # Default start (Kuzu uses 1-based indexing)
        assert expr.args[2] == 5  # stop
        
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "array_slice(n.data, 1, 5)"
    
    def test_list_slicing_with_none_stop(self):
        """Test list slicing with None stop ([2:])."""
        list_field = QueryField("values")
        expr = list_field[2:]
        
        assert isinstance(expr, FunctionExpression)
        assert expr.function_name == "array_slice"
        assert expr.args[0] == list_field
        assert expr.args[1] == 2  # start
        assert expr.args[2] == -1  # Default stop (end of list)
        
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "array_slice(n.values, 2, -1)"
    
    def test_string_slicing(self):
        """Test string slicing using [start:end] operator."""
        string_field = QueryField("description")
        expr = string_field[0:10]
        
        assert isinstance(expr, FunctionExpression)
        assert expr.function_name == "array_slice"
        assert expr.args[0] == string_field
        assert expr.args[1] == 0  # start
        assert expr.args[2] == 10  # stop
        
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "array_slice(n.description, 0, 10)"
    
    # ============================================================================
    # LIST/STRING INDEXING OPERATOR TESTS ([index])
    # ============================================================================
    
    def test_list_indexing_basic(self):
        """Test list indexing operator [index] - Basic functionality."""
        list_field = QueryField("items")
        expr = list_field[2]
        
        # Validate expression type and structure
        assert isinstance(expr, FunctionExpression)
        assert expr.function_name == "array_extract"
        assert expr.args[0] == list_field
        assert expr.args[1] == 2  # index
        
        # Validate Cypher generation
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "array_extract(n.items, 2)"
    
    def test_string_indexing(self):
        """Test string indexing using [index] operator."""
        string_field = QueryField("name")
        expr = string_field[0]
        
        assert isinstance(expr, FunctionExpression)
        assert expr.function_name == "array_extract"
        assert expr.args[0] == string_field
        assert expr.args[1] == 0  # index
        
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "array_extract(n.name, 0)"
    
    def test_dynamic_indexing(self):
        """Test dynamic indexing with variable index."""
        list_field = QueryField("data")
        index_field = QueryField("position")
        expr = list_field[index_field]
        
        assert isinstance(expr, FunctionExpression)
        assert expr.function_name == "array_extract"
        assert expr.args[0] == list_field
        assert expr.args[1] == index_field
        
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "array_extract(n.data, n.position)"
    
    # ============================================================================
    # PATTERN MATCHING OPERATOR TESTS (=~ and !~)
    # ============================================================================
    
    def test_regex_match_method(self):
        """Test regex match using regex_match method (=~ equivalent)."""
        string_field = QueryField("email")
        pattern = r".*@example\.com$"
        expr = string_field.regex_match(pattern)
        
        # Validate expression type and structure
        assert isinstance(expr, FieldFilterExpression)
        assert expr.field_path == "email"
        assert expr.operator == ComparisonOperator.REGEX_MATCH
        assert expr.value == pattern
        
        # Validate Cypher generation
        cypher = expr.to_cypher({"n": "n"})
        params = expr.get_parameters()
        param_name = list(params.keys())[0]
        assert cypher == f"n.email =~ ${param_name}"
    
    def test_not_regex_match_method(self):
        """Test negative regex match using not_regex_match method (!~ equivalent)."""
        string_field = QueryField("username")
        pattern = r"^admin.*"
        expr = string_field.not_regex_match(pattern)
        
        assert isinstance(expr, FieldFilterExpression)
        assert expr.field_path == "username"
        assert expr.operator == ComparisonOperator.NOT_REGEX_MATCH
        assert expr.value == pattern
        
        cypher = expr.to_cypher({"n": "n"})
        params = expr.get_parameters()
        param_name = list(params.keys())[0]
        assert cypher == f"n.username !~ ${param_name}"
    
    def test_pattern_matching_operator_overload(self):
        """Test pattern matching using @ operator (=~ equivalent)."""
        string_field = QueryField("phone")
        pattern = r"^\+1-\d{3}-\d{3}-\d{4}$"
        expr = string_field @ pattern
        
        assert isinstance(expr, FieldFilterExpression)
        assert expr.field_path == "phone"
        assert expr.operator == ComparisonOperator.REGEX_MATCH
        assert expr.value == pattern
        
        cypher = expr.to_cypher({"n": "n"})
        params = expr.get_parameters()
        param_name = list(params.keys())[0]
        assert cypher == f"n.phone =~ ${param_name}"
    
    # ============================================================================
    # COMPLEX COMBINATIONS AND EDGE CASES
    # ============================================================================
    
    def test_nested_list_operations(self):
        """Test nested list operations combining slicing and concatenation."""
        list_field = QueryField("data")
        
        # Test slicing then concatenation: data[1:3] + [10, 20]
        sliced = list_field[1:3]
        additional = [10, 20]
        expr = sliced + additional
        
        cypher = expr.to_cypher({"n": "n"})
        params = expr.get_parameters()
        # ArithmeticExpression uses arith_right_ prefix for right operand parameters
        param_name = [k for k in params.keys() if k.startswith('arith_right_')][0]
        assert cypher == f"(array_slice(n.data, 1, 3) + ${param_name})"
    
    def test_list_concatenation_with_indexing(self):
        """Test list concatenation combined with indexing."""
        list_field1 = QueryField("first")
        list_field2 = QueryField("second")
        
        # Test (first + second)[0]
        concatenated = list_field1 + list_field2
        indexed = concatenated[0]
        
        cypher = indexed.to_cypher({"n": "n"})
        assert cypher == "array_extract((n.first + n.second), 0)"
    
    def test_string_operations_combination(self):
        """Test combination of string operations."""
        string_field = QueryField("text")
        
        # Test string slicing with pattern matching
        sliced = string_field[0:5]
        pattern_match = sliced.regex_match(r"^[A-Z].*")
        
        cypher = pattern_match.to_cypher({"n": "n"})
        params = pattern_match.get_parameters()
        param_name = list(params.keys())[0]
        assert cypher == f"array_slice(n.text, 0, 5) =~ ${param_name}"
    
    def test_parameter_handling_in_list_operations(self):
        """Test parameter handling in list operations."""
        list_field = QueryField("items")
        dynamic_list = ["dynamic", "values"]
        
        expr = list_field + dynamic_list
        params = expr.get_parameters()
        
        # Should have one parameter for the dynamic list
        assert len(params) == 1
        assert dynamic_list in params.values()
    
    def test_field_reference_extraction_list_ops(self):
        """Test field reference extraction from list operations."""
        list_field1 = QueryField("first_list")
        list_field2 = QueryField("second_list")
        
        # Test concatenation
        concat_expr = list_field1 + list_field2
        refs = concat_expr.get_field_references()
        assert "first_list" in refs
        assert "second_list" in refs
        assert len(refs) == 2
        
        # Test slicing
        slice_expr = list_field1[1:3]
        refs = slice_expr.get_field_references()
        assert "first_list" in refs
        assert len(refs) == 1
        
        # Test indexing
        index_expr = list_field1[0]
        refs = index_expr.get_field_references()
        assert "first_list" in refs
        assert len(refs) == 1
    
    def test_all_list_string_operators_combined(self):
        """Test covering all list/string operator types."""
        # List operations
        list_field = QueryField("my_list")
        string_field = QueryField("my_string")
        
        # List concatenation
        concat_expr = list_field + [1, 2, 3]
        
        # List slicing
        slice_expr = list_field[1:5]
        
        # List indexing
        index_expr = list_field[2]
        
        # String slicing
        string_slice_expr = string_field[0:10]
        
        # String indexing
        string_index_expr = string_field[0]
        
        # Pattern matching
        regex_expr = string_field.regex_match(r"^test.*")
        not_regex_expr = string_field.not_regex_match(r"^admin.*")
        
        # Validate all expressions generate correct Cypher
        # Test concat expression
        cypher = concat_expr.to_cypher({"n": "n"})
        params = concat_expr.get_parameters()
        # ArithmeticExpression uses arith_right_ prefix for right operand parameters
        concat_param = [k for k in params.keys() if k.startswith('arith_right_')][0]
        assert cypher == f"(n.my_list + ${concat_param})"

        # Test slice expression
        assert slice_expr.to_cypher({"n": "n"}) == "array_slice(n.my_list, 1, 5)"

        # Test index expression
        assert index_expr.to_cypher({"n": "n"}) == "array_extract(n.my_list, 2)"

        # Test string slice expression
        assert string_slice_expr.to_cypher({"n": "n"}) == "array_slice(n.my_string, 0, 10)"

        # Test string index expression
        assert string_index_expr.to_cypher({"n": "n"}) == "array_extract(n.my_string, 0)"

        # Test regex expression
        cypher = regex_expr.to_cypher({"n": "n"})
        params = regex_expr.get_parameters()
        regex_param = list(params.keys())[0]
        assert cypher == f"n.my_string =~ ${regex_param}"

        # Test not regex expression
        cypher = not_regex_expr.to_cypher({"n": "n"})
        params = not_regex_expr.get_parameters()
        not_regex_param = list(params.keys())[0]
        assert cypher == f"n.my_string !~ ${not_regex_param}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

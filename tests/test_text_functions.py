"""
Tests for text functions in KuzuAlchemy.
"""

import pytest
from kuzualchemy.kuzu_query_fields import QueryField
from kuzualchemy.kuzu_functions import concat, ws_concat, array_extract, array_slice
from kuzualchemy.kuzu_query_expressions import FunctionExpression


class TestTextFunctions:
    """Test text functions on QueryField objects."""

    def test_concat_method(self):
        """Test concat method on QueryField."""
        field = QueryField("name")
        result = field.concat(" ", "suffix")
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "concat"
        assert len(result.args) == 3
        assert result.args[0] == field
        assert result.args[1] == " "
        assert result.args[2] == "suffix"

    def test_lower_method(self):
        """Test lower method on QueryField."""
        field = QueryField("name")
        result = field.lower()
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "lower"
        assert len(result.args) == 1
        assert result.args[0] == field

    def test_upper_method(self):
        """Test upper method on QueryField."""
        field = QueryField("name")
        result = field.upper()
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "upper"
        assert len(result.args) == 1
        assert result.args[0] == field

    def test_lcase_method(self):
        """Test lcase method (alias for lower)."""
        field = QueryField("name")
        result = field.lcase()
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "lcase"
        assert len(result.args) == 1
        assert result.args[0] == field

    def test_ucase_method(self):
        """Test ucase method (alias for upper)."""
        field = QueryField("name")
        result = field.ucase()
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "ucase"
        assert len(result.args) == 1
        assert result.args[0] == field

    def test_initcap_method(self):
        """Test initcap method."""
        field = QueryField("name")
        result = field.initcap()
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "initcap"
        assert len(result.args) == 1
        assert result.args[0] == field

    def test_left_method(self):
        """Test left method."""
        field = QueryField("name")
        result = field.left(5)
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "left"
        assert len(result.args) == 2
        assert result.args[0] == field
        assert result.args[1] == 5

    def test_right_method(self):
        """Test right method."""
        field = QueryField("name")
        result = field.right(3)
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "right"
        assert len(result.args) == 2
        assert result.args[0] == field
        assert result.args[1] == 3

    def test_substring_method(self):
        """Test substring method."""
        field = QueryField("name")
        result = field.substring(2, 5)
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "substring"
        assert len(result.args) == 3
        assert result.args[0] == field
        assert result.args[1] == 2
        assert result.args[2] == 5

    def test_substr_method(self):
        """Test substr method (alias for substring)."""
        field = QueryField("name")
        result = field.substr(1, 3)
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "substr"
        assert len(result.args) == 3
        assert result.args[0] == field
        assert result.args[1] == 1
        assert result.args[2] == 3

    def test_size_method(self):
        """Test size method."""
        field = QueryField("name")
        result = field.size()
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "size"
        assert len(result.args) == 1
        assert result.args[0] == field

    def test_levenshtein_method(self):
        """Test levenshtein method."""
        field = QueryField("name")
        result = field.levenshtein("test")
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "levenshtein"
        assert len(result.args) == 2
        assert result.args[0] == field
        assert result.args[1] == "test"

    def test_trim_methods(self):
        """Test trim, ltrim, rtrim methods."""
        field = QueryField("name")
        
        # Test trim
        result = field.trim()
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "trim"
        assert len(result.args) == 1
        assert result.args[0] == field
        
        # Test ltrim
        result = field.ltrim()
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "ltrim"
        assert len(result.args) == 1
        assert result.args[0] == field
        
        # Test rtrim
        result = field.rtrim()
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "rtrim"
        assert len(result.args) == 1
        assert result.args[0] == field

    def test_pad_methods(self):
        """Test lpad and rpad methods."""
        field = QueryField("name")
        
        # Test lpad
        result = field.lpad(10, "*")
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "lpad"
        assert len(result.args) == 3
        assert result.args[0] == field
        assert result.args[1] == 10
        assert result.args[2] == "*"
        
        # Test rpad
        result = field.rpad(8, "-")
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "rpad"
        assert len(result.args) == 3
        assert result.args[0] == field
        assert result.args[1] == 8
        assert result.args[2] == "-"

    def test_repeat_method(self):
        """Test repeat method."""
        field = QueryField("name")
        result = field.repeat(3)
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "repeat"
        assert len(result.args) == 2
        assert result.args[0] == field
        assert result.args[1] == 3

    def test_reverse_method(self):
        """Test reverse method."""
        field = QueryField("name")
        result = field.reverse()
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "reverse"
        assert len(result.args) == 1
        assert result.args[0] == field

    def test_string_split_method(self):
        """Test string_split method."""
        field = QueryField("name")
        result = field.string_split(",")
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "string_split"
        assert len(result.args) == 2
        assert result.args[0] == field
        assert result.args[1] == ","

    def test_split_part_method(self):
        """Test split_part method."""
        field = QueryField("name")
        result = field.split_part(",", 2)
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "split_part"
        assert len(result.args) == 3
        assert result.args[0] == field
        assert result.args[1] == ","
        assert result.args[2] == 2


class TestStandaloneTextFunctions:
    """Test standalone text functions."""

    def test_concat_function(self):
        """Test standalone concat function."""
        result = concat("hello", " ", "world")
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "concat"
        assert len(result.args) == 3
        assert result.args[0] == "hello"
        assert result.args[1] == " "
        assert result.args[2] == "world"

    def test_ws_concat_function(self):
        """Test ws_concat function."""
        result = ws_concat(",", "a", "b", "c")
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "ws_concat"
        assert len(result.args) == 4
        assert result.args[0] == ","
        assert result.args[1] == "a"
        assert result.args[2] == "b"
        assert result.args[3] == "c"

    def test_array_extract_function(self):
        """Test array_extract function."""
        result = array_extract("test", 2)
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "array_extract"
        assert len(result.args) == 2
        assert result.args[0] == "test"
        assert result.args[1] == 2

    def test_array_slice_function(self):
        """Test array_slice function."""
        result = array_slice("test", 1, 3)
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "array_slice"
        assert len(result.args) == 3
        assert result.args[0] == "test"
        assert result.args[1] == 1
        assert result.args[2] == 3


class TestCypherGeneration:
    """Test Cypher generation for text functions."""

    def test_function_cypher_generation(self):
        """Test that functions generate correct Cypher."""
        field = QueryField("name")
        result = field.upper()
        
        alias_map = {"n": "n"}
        cypher = result.to_cypher(alias_map)
        
        assert cypher == "upper(n.name)"

    def test_function_with_parameters_cypher(self):
        """Test function with parameters generates correct Cypher."""
        field = QueryField("name")
        result = field.substring(2, 5)
        
        alias_map = {"n": "n"}
        cypher = result.to_cypher(alias_map)
        
        assert cypher == "substring(n.name, 2, 5)"

    def test_function_with_string_parameters(self):
        """Test function with string parameters."""
        field = QueryField("name")
        result = field.lpad(10, "*")
        
        alias_map = {"n": "n"}
        cypher = result.to_cypher(alias_map)
        params = result.get_parameters()
        
        assert "lpad(n.name, 10, $" in cypher
        assert len(params) == 1
        assert "*" in params.values()

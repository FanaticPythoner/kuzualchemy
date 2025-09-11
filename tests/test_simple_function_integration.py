# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
Simple integration tests for KuzuAlchemy functions.
"""

from kuzualchemy.kuzu_query_fields import QueryField
from kuzualchemy.kuzu_query_expressions import FunctionExpression


class TestSimpleFunctionIntegration:
    """Simple tests for function creation and Cypher generation."""

    def test_text_function_cypher_generation(self):
        """Test that text functions generate correct Cypher."""
        field = QueryField("name")
        
        # Test upper function
        upper_expr = field.upper()
        assert isinstance(upper_expr, FunctionExpression)
        assert upper_expr.function_name == "upper"
        
        cypher = upper_expr.to_cypher({"n": "n"})
        assert cypher == "upper(n.name)"

    def test_numeric_function_cypher_generation(self):
        """Test that numeric functions generate correct Cypher."""
        field = QueryField("age")
        
        # Test abs function
        abs_expr = field.abs()
        assert isinstance(abs_expr, FunctionExpression)
        assert abs_expr.function_name == "abs"
        
        cypher = abs_expr.to_cypher({"n": "n"})
        assert cypher == "abs(n.age)"

    def test_function_with_parameters_cypher(self):
        """Test functions with parameters generate correct Cypher."""
        field = QueryField("name")
        
        # Test substring function
        substr_expr = field.substring(1, 5)
        assert isinstance(substr_expr, FunctionExpression)
        assert substr_expr.function_name == "substring"
        
        cypher = substr_expr.to_cypher({"n": "n"})
        assert cypher == "substring(n.name, 1, 5)"

    def test_list_function_cypher_generation(self):
        """Test that list functions generate correct Cypher."""
        field = QueryField("tags")
        
        # Test list_append function
        append_expr = field.list_append("new_tag")
        assert isinstance(append_expr, FunctionExpression)
        assert append_expr.function_name == "list_append"
        
        cypher = append_expr.to_cypher({"n": "n"})
        params = append_expr.get_parameters()
        
        assert "list_append(n.tags, $" in cypher
        assert "new_tag" in params.values()

    def test_chained_functions_cypher(self):
        """Test chaining functions generates correct Cypher."""
        field = QueryField("name")
        
        # Chain upper and substring
        chained_expr = field.upper().substring(1, 3)
        
        cypher = chained_expr.to_cypher({"n": "n"})
        assert cypher == "substring(upper(n.name), 1, 3)"

    def test_standalone_function_cypher(self):
        """Test standalone functions generate correct Cypher."""
        from kuzualchemy.kuzu_functions import pi, concat
        
        # Test pi function
        pi_expr = pi()
        cypher = pi_expr.to_cypher({"n": "n"})
        assert cypher == "pi()"
        
        # Test concat function
        concat_expr = concat("Hello", " ", "World")
        cypher = concat_expr.to_cypher({"n": "n"})
        params = concat_expr.get_parameters()
        
        assert "concat($" in cypher
        assert "Hello" in params.values()
        assert " " in params.values()
        assert "World" in params.values()

    def test_function_parameters_handling(self):
        """Test that function parameters are handled correctly."""
        field = QueryField("name")
        
        # Test function with string parameter
        lpad_expr = field.lpad(10, "*")
        params = lpad_expr.get_parameters()
        
        # Should have parameters for count and character
        assert len(params) == 1  # Only the character parameter (count is literal)
        assert "*" in params.values()

    def test_function_field_references(self):
        """Test that function field references are tracked correctly."""
        field1 = QueryField("name")
        field2 = QueryField("age")
        
        # Test function with multiple field references
        concat_expr = field1.concat(" is ", field2, " years old")
        refs = concat_expr.get_field_references()
        
        assert "name" in refs
        assert "age" in refs

    def test_arithmetic_on_functions(self):
        """Test arithmetic operations on function results."""
        field = QueryField("value")
        
        # Test arithmetic on function result
        expr = field.sqrt() + 5
        
        cypher = expr.to_cypher({"n": "n"})
        assert "sqrt(n.value) + 5" in cypher

    def test_comparison_on_functions(self):
        """Test comparison operations on function results."""
        field = QueryField("name")
        
        # Test comparison on function result
        expr = field.size() > 10
        
        cypher = expr.to_cypher({"n": "n"})
        assert "size(n.name) > 10" in cypher

    def test_pattern_matching_on_functions(self):
        """Test pattern matching on function results."""
        field = QueryField("name")
        
        # Test regex on function result
        expr = field.upper().regex_match("^[A-Z]+$")
        
        cypher = expr.to_cypher({"n": "n"})
        params = expr.get_parameters()
        
        assert "upper(n.name) =~ $" in cypher
        assert "^[A-Z]+$" in params.values()

    def test_nested_function_calls(self):
        """Test deeply nested function calls."""
        field = QueryField("text")
        
        # Create a complex nested expression
        expr = field.trim().upper().substring(1, 5).size()
        
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "size(substring(upper(trim(n.text)), 1, 5))"

    def test_function_aliases(self):
        """Test that function aliases work correctly."""
        field = QueryField("name")
        
        # Test lcase (alias for lower)
        lcase_expr = field.lcase()
        assert lcase_expr.function_name == "lcase"
        
        # Test substr (alias for substring)
        substr_expr = field.substr(1, 3)
        assert substr_expr.function_name == "substr"
        
        # Test ceiling (alias for ceil)
        field_num = QueryField("value")
        ceiling_expr = field_num.ceiling()
        assert ceiling_expr.function_name == "ceiling"

    def test_date_functions(self):
        """Test date functions generate correct Cypher."""
        field = QueryField("created_date")
        
        # Test date_part function
        date_part_expr = field.date_part("year")
        cypher = date_part_expr.to_cypher({"n": "n"})
        params = date_part_expr.get_parameters()
        
        assert "date_part($" in cypher
        assert "year" in params.values()

    def test_timestamp_functions(self):
        """Test timestamp functions generate correct Cypher."""
        field = QueryField("created_at")
        
        # Test century function
        century_expr = field.century()
        cypher = century_expr.to_cypher({"n": "n"})
        assert cypher == "century(n.created_at)"

    def test_regex_functions(self):
        """Test regex functions generate correct Cypher."""
        field = QueryField("text")
        
        # Test regexp_replace function
        replace_expr = field.regexp_replace("[0-9]+", "X")
        cypher = replace_expr.to_cypher({"n": "n"})
        params = replace_expr.get_parameters()
        
        assert "regexp_replace(n.text, $" in cypher
        assert "[0-9]+" in params.values()
        assert "X" in params.values()

    def test_function_error_handling(self):
        """Test that function creation handles edge cases."""
        field = QueryField("value")
        
        # Test function with no arguments
        size_expr = field.size()
        assert len(size_expr.args) == 1
        assert size_expr.args[0] == field
        
        # Test function with multiple arguments
        substr_expr = field.substring(1, 5)
        assert len(substr_expr.args) == 3
        assert substr_expr.args[0] == field
        assert substr_expr.args[1] == 1
        assert substr_expr.args[2] == 5

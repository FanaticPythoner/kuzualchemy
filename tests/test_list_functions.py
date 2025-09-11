# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
Tests for list functions in KuzuAlchemy.
"""

from kuzualchemy.kuzu_query_fields import QueryField
from kuzualchemy.kuzu_functions import list_creation, size, list_concat, range
from kuzualchemy.kuzu_query_expressions import FunctionExpression


class TestListFunctions:
    """Test list functions on QueryField objects."""

    def test_list_concat_method(self):
        """Test list_concat method on QueryField."""
        field = QueryField("items")
        result = field.list_concat([4, 5, 6])
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "list_concat"
        assert len(result.args) == 2
        assert result.args[0] == field
        assert result.args[1] == [4, 5, 6]

    def test_list_cat_method(self):
        """Test list_cat method (alias for list_concat)."""
        field = QueryField("items")
        result = field.list_cat([7, 8])
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "list_cat"
        assert len(result.args) == 2
        assert result.args[0] == field
        assert result.args[1] == [7, 8]

    def test_array_concat_method(self):
        """Test array_concat method (alias for list_concat)."""
        field = QueryField("items")
        result = field.array_concat([9, 10])
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "array_concat"
        assert len(result.args) == 2
        assert result.args[0] == field
        assert result.args[1] == [9, 10]

    def test_array_cat_method(self):
        """Test array_cat method (alias for list_concat)."""
        field = QueryField("items")
        result = field.array_cat([11, 12])
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "array_cat"
        assert len(result.args) == 2
        assert result.args[0] == field
        assert result.args[1] == [11, 12]

    def test_list_append_method(self):
        """Test list_append method."""
        field = QueryField("items")
        result = field.list_append(42)
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "list_append"
        assert len(result.args) == 2
        assert result.args[0] == field
        assert result.args[1] == 42

    def test_array_append_method(self):
        """Test array_append method (alias for list_append)."""
        field = QueryField("items")
        result = field.array_append(43)
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "array_append"
        assert len(result.args) == 2
        assert result.args[0] == field
        assert result.args[1] == 43

    def test_array_push_back_method(self):
        """Test array_push_back method (alias for list_append)."""
        field = QueryField("items")
        result = field.array_push_back(44)
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "array_push_back"
        assert len(result.args) == 2
        assert result.args[0] == field
        assert result.args[1] == 44

    def test_list_prepend_method(self):
        """Test list_prepend method."""
        field = QueryField("items")
        result = field.list_prepend(0)
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "list_prepend"
        assert len(result.args) == 2
        assert result.args[0] == field
        assert result.args[1] == 0

    def test_array_prepend_method(self):
        """Test array_prepend method (alias for list_prepend)."""
        field = QueryField("items")
        result = field.array_prepend(-1)
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "array_prepend"
        assert len(result.args) == 2
        assert result.args[0] == field
        assert result.args[1] == -1

    def test_array_push_front_method(self):
        """Test array_push_front method (alias for list_prepend)."""
        field = QueryField("items")
        result = field.array_push_front(-2)
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "array_push_front"
        assert len(result.args) == 2
        assert result.args[0] == field
        assert result.args[1] == -2

    def test_list_position_method(self):
        """Test list_position method."""
        field = QueryField("items")
        result = field.list_position(5)
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "list_position"
        assert len(result.args) == 2
        assert result.args[0] == field
        assert result.args[1] == 5

    def test_list_indexof_method(self):
        """Test list_indexof method (alias for list_position)."""
        field = QueryField("items")
        result = field.list_indexof(6)
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "list_indexof"
        assert len(result.args) == 2
        assert result.args[0] == field
        assert result.args[1] == 6

    def test_array_position_method(self):
        """Test array_position method (alias for list_position)."""
        field = QueryField("items")
        result = field.array_position(7)
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "array_position"
        assert len(result.args) == 2
        assert result.args[0] == field
        assert result.args[1] == 7

    def test_array_indexof_method(self):
        """Test array_indexof method (alias for list_position)."""
        field = QueryField("items")
        result = field.array_indexof(8)
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "array_indexof"
        assert len(result.args) == 2
        assert result.args[0] == field
        assert result.args[1] == 8

    def test_list_contains_method(self):
        """Test list_contains method."""
        field = QueryField("items")
        result = field.list_contains(9)
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "list_contains"
        assert len(result.args) == 2
        assert result.args[0] == field
        assert result.args[1] == 9

    def test_list_has_method(self):
        """Test list_has method (alias for list_contains)."""
        field = QueryField("items")
        result = field.list_has(10)
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "list_has"
        assert len(result.args) == 2
        assert result.args[0] == field
        assert result.args[1] == 10

    def test_array_contains_method(self):
        """Test array_contains method (alias for list_contains)."""
        field = QueryField("items")
        result = field.array_contains(11)
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "array_contains"
        assert len(result.args) == 2
        assert result.args[0] == field
        assert result.args[1] == 11

    def test_array_has_method(self):
        """Test array_has method (alias for list_contains)."""
        field = QueryField("items")
        result = field.array_has(12)
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "array_has"
        assert len(result.args) == 2
        assert result.args[0] == field
        assert result.args[1] == 12

    def test_list_slice_method(self):
        """Test list_slice method."""
        field = QueryField("items")
        result = field.list_slice(2, 5)
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "list_slice"
        assert len(result.args) == 3
        assert result.args[0] == field
        assert result.args[1] == 2
        assert result.args[2] == 5

    def test_list_reverse_method(self):
        """Test list_reverse method."""
        field = QueryField("items")
        result = field.list_reverse()
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "list_reverse"
        assert len(result.args) == 1
        assert result.args[0] == field

    def test_list_sort_method(self):
        """Test list_sort method."""
        field = QueryField("items")
        
        # Test default ascending sort
        result = field.list_sort()
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "list_sort"
        assert len(result.args) == 1
        assert result.args[0] == field
        
        # Test descending sort
        result = field.list_sort("DESC")
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "list_sort"
        assert len(result.args) == 2
        assert result.args[0] == field
        assert result.args[1] == "DESC"

    def test_list_reverse_sort_method(self):
        """Test list_reverse_sort method."""
        field = QueryField("items")
        result = field.list_reverse_sort()
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "list_reverse_sort"
        assert len(result.args) == 1
        assert result.args[0] == field

    def test_list_aggregation_methods(self):
        """Test list aggregation methods."""
        field = QueryField("items")
        
        # Test list_sum
        result = field.list_sum()
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "list_sum"
        assert len(result.args) == 1
        assert result.args[0] == field
        
        # Test list_product
        result = field.list_product()
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "list_product"
        assert len(result.args) == 1
        assert result.args[0] == field

    def test_list_utility_methods(self):
        """Test list utility methods."""
        field = QueryField("items")
        
        # Test list_distinct
        result = field.list_distinct()
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "list_distinct"
        assert len(result.args) == 1
        assert result.args[0] == field
        
        # Test list_unique
        result = field.list_unique()
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "list_unique"
        assert len(result.args) == 1
        assert result.args[0] == field
        
        # Test list_any_value
        result = field.list_any_value()
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "list_any_value"
        assert len(result.args) == 1
        assert result.args[0] == field

    def test_list_to_string_method(self):
        """Test list_to_string method."""
        field = QueryField("items")
        result = field.list_to_string(",")
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "list_to_string"
        assert len(result.args) == 2
        assert result.args[0] == ","
        assert result.args[1] == field


class TestStandaloneListFunctions:
    """Test standalone list functions."""

    def test_list_creation_function(self):
        """Test list_creation function."""
        result = list_creation(1, 2, 3, 4, 5)
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "list_creation"
        assert len(result.args) == 5
        assert result.args == [1, 2, 3, 4, 5]

    def test_size_function(self):
        """Test size function."""
        result = size([1, 2, 3])
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "size"
        assert len(result.args) == 1
        assert result.args[0] == [1, 2, 3]

    def test_list_concat_function(self):
        """Test standalone list_concat function."""
        result = list_concat([1, 2], [3, 4])
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "list_concat"
        assert len(result.args) == 2
        assert result.args[0] == [1, 2]
        assert result.args[1] == [3, 4]

    def test_range_function(self):
        """Test range function."""
        # Test with default step
        result = range(1, 5)
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "range"
        assert len(result.args) == 2
        assert result.args[0] == 1
        assert result.args[1] == 5
        
        # Test with custom step
        result = range(1, 10, 2)
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "range"
        assert len(result.args) == 3
        assert result.args[0] == 1
        assert result.args[1] == 10
        assert result.args[2] == 2

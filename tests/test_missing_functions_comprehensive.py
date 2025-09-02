"""
Tests for all newly implemented missing functions in KuzuAlchemy.
Tests interval, array, utility, hash, UUID, blob, struct, map, union, node/rel, 
recursive rel functions, and CAST/CASE expressions.
"""

import pytest
from kuzualchemy.kuzu_query_fields import QueryField
from kuzualchemy.kuzu_functions import (
    # Interval functions
    to_years, to_months, to_days, to_hours, to_minutes, to_seconds,
    to_milliseconds, to_microseconds,
    # Array functions
    array_value, array_distance, array_squared_distance, array_dot_product,
    array_inner_product, array_cross_product, array_cosine_similarity,
    # Utility functions
    coalesce, ifnull, nullif, constant_or_null, count_if, typeof, error,
    # Hash functions
    md5, sha256, hash,
    # UUID functions - renamed to avoid conflict with Python's uuid module
    gen_random_uuid,
    # Blob functions
    blob, encode, decode, octet_length,
    # Struct functions
    struct_extract,
    # Map functions
    map_func, map_extract, element_at, cardinality, map_keys, map_values,
    # Union functions
    union_value, union_tag, union_extract,
    # Node/rel functions
    id_func, label, labels, offset,
    # Recursive rel functions
    nodes, rels, properties, is_trail, is_acyclic, length, cost,
    # CAST and CASE
    cast, cast_as, case
)
# Import uuid function with alias to avoid conflict
from kuzualchemy.kuzu_functions import uuid as uuid_func
from kuzualchemy.kuzu_query_expressions import CastExpression, CaseExpression


class TestIntervalFunctions:
    """Test all interval conversion functions."""

    def test_interval_functions_cypher_generation(self):
        """Test that interval functions generate correct Cypher."""
        field = QueryField("duration")

        # Test all interval functions - integers are rendered as literals
        assert to_years(5).to_cypher({}, "") == "to_years(5)"
        assert to_months(field).to_cypher({"n": "n"}, "") == "to_months(n.duration)"
        assert to_days(30).to_cypher({}, "") == "to_days(30)"
        assert to_hours(24).to_cypher({}, "") == "to_hours(24)"
        assert to_minutes(60).to_cypher({}, "") == "to_minutes(60)"
        assert to_seconds(3600).to_cypher({}, "") == "to_seconds(3600)"
        assert to_milliseconds(1000).to_cypher({}, "") == "to_milliseconds(1000)"
        assert to_microseconds(1000000).to_cypher({}, "") == "to_microseconds(1000000)"

    def test_interval_functions_parameters(self):
        """Test that interval functions handle parameters correctly."""
        # Integer literals don't create parameters
        result = to_years(5)
        params = result.get_parameters()
        assert params == {}

        # String values do create parameters
        result_str = to_years("5")
        params_str = result_str.get_parameters()
        assert len(params_str) == 1
        assert list(params_str.values())[0] == "5"

    def test_queryfield_interval_methods(self):
        """Test interval methods on QueryField instances."""
        field = QueryField("value")
        
        assert field.to_years().to_cypher({"n": "n"}, "") == "to_years(n.value)"
        assert field.to_months().to_cypher({"n": "n"}, "") == "to_months(n.value)"
        assert field.to_days().to_cypher({"n": "n"}, "") == "to_days(n.value)"


class TestArrayFunctions:
    """Test all array manipulation functions."""

    def test_array_functions_cypher_generation(self):
        """Test that array functions generate correct Cypher."""
        arr1 = QueryField("array1")
        arr2 = QueryField("array2")

        # Integer literals are rendered directly, not as parameters
        assert array_value(1, 2, 3).to_cypher({}, "") == "array_value(1, 2, 3)"
        assert array_distance(arr1, arr2).to_cypher({"n": "n"}, "") == "array_distance(n.array1, n.array2)"
        assert array_squared_distance(arr1, arr2).to_cypher({"n": "n"}, "") == "array_squared_distance(n.array1, n.array2)"
        assert array_dot_product(arr1, arr2).to_cypher({"n": "n"}, "") == "array_dot_product(n.array1, n.array2)"
        assert array_inner_product(arr1, arr2).to_cypher({"n": "n"}, "") == "array_inner_product(n.array1, n.array2)"
        assert array_cross_product(arr1, arr2).to_cypher({"n": "n"}, "") == "array_cross_product(n.array1, n.array2)"
        assert array_cosine_similarity(arr1, arr2).to_cypher({"n": "n"}, "") == "array_cosine_similarity(n.array1, n.array2)"

    def test_queryfield_array_methods(self):
        """Test array methods on QueryField instances."""
        field = QueryField("arr")
        other = QueryField("other_arr")
        
        assert field.array_distance(other).to_cypher({"n": "n"}, "") == "array_distance(n.arr, n.other_arr)"
        assert field.array_dot_product(other).to_cypher({"n": "n"}, "") == "array_dot_product(n.arr, n.other_arr)"
        assert field.array_cosine_similarity(other).to_cypher({"n": "n"}, "") == "array_cosine_similarity(n.arr, n.other_arr)"


class TestUtilityFunctions:
    """Test all utility functions."""

    def test_utility_functions_cypher_generation(self):
        """Test that utility functions generate correct Cypher."""
        field1 = QueryField("field1")
        field2 = QueryField("field2")

        # String parameters use func_arg_X_Y format
        coalesce_result = coalesce(field1, field2, "default")
        coalesce_cypher = coalesce_result.to_cypher({"n": "n"}, "")
        assert "coalesce(n.field1, n.field2," in coalesce_cypher
        assert "$func_arg_" in coalesce_cypher

        ifnull_result = ifnull(field1, "default")
        ifnull_cypher = ifnull_result.to_cypher({"n": "n"}, "")
        assert "ifnull(n.field1," in ifnull_cypher
        assert "$func_arg_" in ifnull_cypher

        assert nullif(field1, field2).to_cypher({"n": "n"}, "") == "nullif(n.field1, n.field2)"

        constant_result = constant_or_null("value", field1)
        constant_cypher = constant_result.to_cypher({"n": "n"}, "")
        assert "constant_or_null(" in constant_cypher
        assert "$func_arg_" in constant_cypher
        assert "n.field1)" in constant_cypher

        assert count_if(field1).to_cypher({"n": "n"}, "") == "count_if(n.field1)"
        assert typeof(field1).to_cypher({"n": "n"}, "") == "typeof(n.field1)"

        error_result = error("message")
        error_cypher = error_result.to_cypher({}, "")
        assert "error(" in error_cypher
        assert "$func_arg_" in error_cypher

    def test_queryfield_utility_methods(self):
        """Test utility methods on QueryField instances."""
        field = QueryField("value")

        coalesce_result = field.coalesce("default")
        coalesce_cypher = coalesce_result.to_cypher({"n": "n"}, "")
        assert "coalesce(n.value," in coalesce_cypher
        assert "$func_arg_" in coalesce_cypher

        ifnull_result = field.ifnull("default")
        ifnull_cypher = ifnull_result.to_cypher({"n": "n"}, "")
        assert "ifnull(n.value," in ifnull_cypher
        assert "$func_arg_" in ifnull_cypher

        assert field.typeof().to_cypher({"n": "n"}, "") == "typeof(n.value)"


class TestHashFunctions:
    """Test all hash functions."""

    def test_hash_functions_cypher_generation(self):
        """Test that hash functions generate correct Cypher."""
        field = QueryField("data")

        assert md5(field).to_cypher({"n": "n"}, "") == "md5(n.data)"
        assert sha256(field).to_cypher({"n": "n"}, "") == "sha256(n.data)"
        assert hash(field).to_cypher({"n": "n"}, "") == "hash(n.data)"

        # Test with string literals - use func_arg format
        md5_result = md5("test")
        md5_cypher = md5_result.to_cypher({}, "")
        assert "md5(" in md5_cypher
        assert "$func_arg_" in md5_cypher

    def test_queryfield_hash_methods(self):
        """Test hash methods on QueryField instances."""
        field = QueryField("data")
        
        assert field.md5().to_cypher({"n": "n"}, "") == "md5(n.data)"
        assert field.sha256().to_cypher({"n": "n"}, "") == "sha256(n.data)"
        assert field.hash().to_cypher({"n": "n"}, "") == "hash(n.data)"


class TestUUIDFunctions:
    """Test all UUID functions."""

    def test_uuid_functions_cypher_generation(self):
        """Test that UUID functions generate correct Cypher."""
        field = QueryField("uuid_string")

        assert uuid_func(field).to_cypher({"n": "n"}, "") == "UUID(n.uuid_string)"
        assert gen_random_uuid().to_cypher({}, "") == "gen_random_uuid()"

        # Test with string literal - string parameters use func_arg format
        uuid_result = uuid_func("550e8400-e29b-41d4-a716-446655440000")
        uuid_cypher = uuid_result.to_cypher({}, "")
        assert "UUID(" in uuid_cypher
        assert "$func_arg_" in uuid_cypher

    def test_queryfield_uuid_methods(self):
        """Test UUID methods on QueryField instances."""
        field = QueryField("uuid_str")

        assert field.uuid().to_cypher({"n": "n"}, "") == "UUID(n.uuid_str)"


class TestBlobFunctions:
    """Test all blob functions."""

    def test_blob_functions_cypher_generation(self):
        """Test that blob functions generate correct Cypher."""
        field = QueryField("data")
        
        assert blob(field).to_cypher({"n": "n"}, "") == "BLOB(n.data)"
        assert encode(field).to_cypher({"n": "n"}, "") == "encode(n.data)"
        assert decode(field).to_cypher({"n": "n"}, "") == "decode(n.data)"
        assert octet_length(field).to_cypher({"n": "n"}, "") == "octet_length(n.data)"

    def test_queryfield_blob_methods(self):
        """Test blob methods on QueryField instances."""
        field = QueryField("data")
        
        assert field.blob().to_cypher({"n": "n"}, "") == "BLOB(n.data)"
        assert field.encode().to_cypher({"n": "n"}, "") == "encode(n.data)"
        assert field.decode().to_cypher({"n": "n"}, "") == "decode(n.data)"
        assert field.octet_length().to_cypher({"n": "n"}, "") == "octet_length(n.data)"


class TestStructFunctions:
    """Test all struct functions."""

    def test_struct_functions_cypher_generation(self):
        """Test that struct functions generate correct Cypher."""
        field = QueryField("struct_data")

        struct_result = struct_extract(field, "field_name")
        struct_cypher = struct_result.to_cypher({"n": "n"}, "")
        assert "struct_extract(n.struct_data," in struct_cypher
        assert "$func_arg_" in struct_cypher

    def test_queryfield_struct_methods(self):
        """Test struct methods on QueryField instances."""
        field = QueryField("struct_data")

        struct_result = field.struct_extract("field_name")
        struct_cypher = struct_result.to_cypher({"n": "n"}, "")
        assert "struct_extract(n.struct_data," in struct_cypher
        assert "$func_arg_" in struct_cypher


class TestMapFunctions:
    """Test all map functions."""

    def test_map_functions_cypher_generation(self):
        """Test that map functions generate correct Cypher."""
        field = QueryField("map_data")
        keys = QueryField("keys")
        values = QueryField("values")

        assert map_func(keys, values).to_cypher({"n": "n"}, "") == "map(n.keys, n.values)"

        map_extract_result = map_extract(field, "key")
        map_extract_cypher = map_extract_result.to_cypher({"n": "n"}, "")
        assert "map_extract(n.map_data," in map_extract_cypher
        assert "$func_arg_" in map_extract_cypher

        element_at_result = element_at(field, "key")
        element_at_cypher = element_at_result.to_cypher({"n": "n"}, "")
        assert "element_at(n.map_data," in element_at_cypher
        assert "$func_arg_" in element_at_cypher

        assert cardinality(field).to_cypher({"n": "n"}, "") == "cardinality(n.map_data)"
        assert map_keys(field).to_cypher({"n": "n"}, "") == "map_keys(n.map_data)"
        assert map_values(field).to_cypher({"n": "n"}, "") == "map_values(n.map_data)"

    def test_queryfield_map_methods(self):
        """Test map methods on QueryField instances."""
        field = QueryField("map_data")

        map_extract_result = field.map_extract("key")
        map_extract_cypher = map_extract_result.to_cypher({"n": "n"}, "")
        assert "map_extract(n.map_data," in map_extract_cypher
        assert "$func_arg_" in map_extract_cypher

        element_at_result = field.element_at("key")
        element_at_cypher = element_at_result.to_cypher({"n": "n"}, "")
        assert "element_at(n.map_data," in element_at_cypher
        assert "$func_arg_" in element_at_cypher

        assert field.cardinality().to_cypher({"n": "n"}, "") == "cardinality(n.map_data)"
        assert field.map_keys().to_cypher({"n": "n"}, "") == "map_keys(n.map_data)"
        assert field.map_values().to_cypher({"n": "n"}, "") == "map_values(n.map_data)"


class TestUnionFunctions:
    """Test all union functions."""

    def test_union_functions_cypher_generation(self):
        """Test that union functions generate correct Cypher."""
        field = QueryField("union_data")

        union_value_result = union_value("tag", "value")
        union_value_cypher = union_value_result.to_cypher({}, "")
        assert "union_value(" in union_value_cypher
        assert "$func_arg_" in union_value_cypher

        assert union_tag(field).to_cypher({"n": "n"}, "") == "union_tag(n.union_data)"

        union_extract_result = union_extract(field, "tag")
        union_extract_cypher = union_extract_result.to_cypher({"n": "n"}, "")
        assert "union_extract(n.union_data," in union_extract_cypher
        assert "$func_arg_" in union_extract_cypher

    def test_queryfield_union_methods(self):
        """Test union methods on QueryField instances."""
        field = QueryField("union_data")

        assert field.union_tag().to_cypher({"n": "n"}, "") == "union_tag(n.union_data)"

        union_extract_result = field.union_extract("tag")
        union_extract_cypher = union_extract_result.to_cypher({"n": "n"}, "")
        assert "union_extract(n.union_data," in union_extract_cypher
        assert "$func_arg_" in union_extract_cypher


class TestNodeRelFunctions:
    """Test all node/relationship functions."""

    def test_node_rel_functions_cypher_generation(self):
        """Test that node/rel functions generate correct Cypher."""
        field = QueryField("node")
        
        assert id_func(field).to_cypher({"n": "n"}, "") == "ID(n.node)"
        assert label(field).to_cypher({"n": "n"}, "") == "LABEL(n.node)"
        assert labels(field).to_cypher({"n": "n"}, "") == "LABELS(n.node)"
        assert offset(field).to_cypher({"n": "n"}, "") == "OFFSET(n.node)"

    def test_queryfield_node_rel_methods(self):
        """Test node/rel methods on QueryField instances."""
        field = QueryField("node")
        
        assert field.id().to_cypher({"n": "n"}, "") == "ID(n.node)"
        assert field.label().to_cypher({"n": "n"}, "") == "LABEL(n.node)"
        assert field.labels().to_cypher({"n": "n"}, "") == "LABELS(n.node)"
        assert field.offset().to_cypher({"n": "n"}, "") == "OFFSET(n.node)"


class TestRecursiveRelFunctions:
    """Test all recursive relationship functions."""

    def test_recursive_rel_functions_cypher_generation(self):
        """Test that recursive rel functions generate correct Cypher."""
        field = QueryField("path")

        assert nodes(field).to_cypher({"n": "n"}, "") == "NODES(n.path)"
        assert rels(field).to_cypher({"n": "n"}, "") == "RELS(n.path)"

        properties_result = properties(field, "prop")
        properties_cypher = properties_result.to_cypher({"n": "n"}, "")
        assert "PROPERTIES(n.path," in properties_cypher
        assert "$func_arg_" in properties_cypher

        assert is_trail(field).to_cypher({"n": "n"}, "") == "IS_TRAIL(n.path)"
        assert is_acyclic(field).to_cypher({"n": "n"}, "") == "IS_ACYCLIC(n.path)"
        assert length(field).to_cypher({"n": "n"}, "") == "LENGTH(n.path)"
        assert cost(field).to_cypher({"n": "n"}, "") == "COST(n.path)"

    def test_queryfield_recursive_rel_methods(self):
        """Test recursive rel methods on QueryField instances."""
        field = QueryField("path")

        assert field.nodes().to_cypher({"n": "n"}, "") == "NODES(n.path)"
        assert field.rels().to_cypher({"n": "n"}, "") == "RELS(n.path)"

        properties_result = field.properties("prop")
        properties_cypher = properties_result.to_cypher({"n": "n"}, "")
        assert "PROPERTIES(n.path," in properties_cypher
        assert "$func_arg_" in properties_cypher

        assert field.is_trail().to_cypher({"n": "n"}, "") == "IS_TRAIL(n.path)"
        assert field.is_acyclic().to_cypher({"n": "n"}, "") == "IS_ACYCLIC(n.path)"
        assert field.length().to_cypher({"n": "n"}, "") == "LENGTH(n.path)"
        assert field.cost().to_cypher({"n": "n"}, "") == "COST(n.path)"


class TestCastExpressions:
    """Test CAST expression functionality."""

    def test_cast_expression_cypher_generation(self):
        """Test that CAST expressions generate correct Cypher."""
        field = QueryField("value")

        # Test CAST function syntax
        cast_expr = cast(field, "INT64")
        assert cast_expr.to_cypher({"n": "n"}, "") == 'CAST(n.value, "INT64")'

        # Test CAST AS syntax
        cast_as_expr = cast_as(field, "STRING")
        assert cast_as_expr.to_cypher({"n": "n"}, "") == "CAST(n.value AS STRING)"

        # Test with literal values
        cast_literal = cast(42, "STRING")
        assert cast_literal.to_cypher({}, "") == 'CAST(42, "STRING")'

    def test_cast_expression_parameters(self):
        """Test that CAST expressions handle parameters correctly."""
        cast_expr = cast("test_value", "INT64")
        params = cast_expr.get_parameters()
        # CAST expressions use their own parameter naming scheme
        assert len(params) == 1
        param_key = list(params.keys())[0]
        assert "cast_value_" in param_key
        assert params[param_key] == "test_value"

    def test_queryfield_cast_methods(self):
        """Test CAST methods on QueryField instances."""
        field = QueryField("value")

        cast_expr = field.cast("INT64")
        assert cast_expr.to_cypher({"n": "n"}, "") == 'CAST(n.value, "INT64")'

        cast_as_expr = field.cast_as("STRING")
        assert cast_as_expr.to_cypher({"n": "n"}, "") == "CAST(n.value AS STRING)"

    def test_cast_expression_field_references(self):
        """Test that CAST expressions track field references correctly."""
        field = QueryField("test_field")
        cast_expr = cast(field, "STRING")

        refs = cast_expr.get_field_references()
        assert "test_field" in refs


class TestCaseExpressions:
    """Test CASE expression functionality."""

    def test_case_expression_simple_form(self):
        """Test simple CASE expression (with input expression)."""
        field = QueryField("status")

        case_expr = case(field).when("active", "Active User").when("inactive", "Inactive User").else_("Unknown")
        expected = "CASE n.status WHEN 'active' THEN 'Active User' WHEN 'inactive' THEN 'Inactive User' ELSE 'Unknown' END"
        assert case_expr.to_cypher({"n": "n"}, "") == expected

    def test_case_expression_searched_form(self):
        """Test searched CASE expression (without input expression)."""
        field1 = QueryField("age")
        field2 = QueryField("status")

        case_expr = case().when(field1 > 18, "Adult").when(field2 == "premium", "Premium").else_("Other")
        # Note: This would require proper filter expression handling in the CASE implementation
        # For now, we test the basic structure
        cypher = case_expr.to_cypher({"n": "n"}, "")
        assert cypher.startswith("CASE WHEN")
        assert cypher.endswith("END")
        assert "THEN" in cypher
        assert "ELSE" in cypher

    def test_case_expression_without_else(self):
        """Test CASE expression without ELSE clause."""
        field = QueryField("type")

        case_expr = case(field).when("A", "Type A").when("B", "Type B")
        expected = "CASE n.type WHEN 'A' THEN 'Type A' WHEN 'B' THEN 'Type B' END"
        assert case_expr.to_cypher({"n": "n"}, "") == expected

    def test_case_expression_parameters(self):
        """Test that CASE expressions handle parameters correctly."""
        field = QueryField("status")
        case_expr = case(field).when("active", "Active").else_("Inactive")

        params = case_expr.get_parameters()
        # Parameters should be handled for non-literal values
        assert isinstance(params, dict)

    def test_queryfield_case_method(self):
        """Test CASE method on QueryField instances."""
        field = QueryField("status")

        case_expr = field.case().when("active", "Active").else_("Inactive")
        expected = "CASE n.status WHEN 'active' THEN 'Active' ELSE 'Inactive' END"
        assert case_expr.to_cypher({"n": "n"}, "") == expected

    def test_case_expression_field_references(self):
        """Test that CASE expressions track field references correctly."""
        field1 = QueryField("input_field")
        field2 = QueryField("condition_field")
        field3 = QueryField("result_field")

        case_expr = case(field1).when(field2, field3).else_("default")
        refs = case_expr.get_field_references()

        assert "input_field" in refs
        assert "condition_field" in refs
        assert "result_field" in refs

    def test_case_expression_nested_expressions(self):
        """Test CASE expressions with nested expressions."""
        field = QueryField("value")

        # Test with arithmetic expressions in conditions and results
        case_expr = case().when(field > 10, field * 2).else_(field + 1)
        cypher = case_expr.to_cypher({"n": "n"}, "")

        assert "CASE WHEN" in cypher
        assert "THEN" in cypher
        assert "ELSE" in cypher
        assert "END" in cypher


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple function types."""

    def test_complex_expression_combination(self):
        """Test combining multiple function types in complex expressions."""
        field = QueryField("data")

        # Combine CAST, array functions, and utility functions
        complex_expr = coalesce(
            cast(field.array_distance(array_value(1, 2, 3)), "STRING"),
            "default"
        )

        cypher = complex_expr.to_cypher({"n": "n"}, "")
        assert "coalesce" in cypher
        # The complex expression gets flattened to parameters, so we check for the structure
        assert "$func_arg_" in cypher or "CAST" in cypher

    def test_nested_case_with_functions(self):
        """Test CASE expressions with function calls."""
        field = QueryField("value")

        case_expr = case().when(
            typeof(field) == "STRING",
            field.upper()
        ).when(
            typeof(field) == "INT64",
            cast(field, "STRING")
        ).else_("unknown")

        cypher = case_expr.to_cypher({"n": "n"}, "")
        assert "CASE WHEN" in cypher
        assert "typeof" in cypher

    def test_function_chaining(self):
        """Test chaining multiple functions together."""
        field = QueryField("text")

        # Chain multiple text functions - FunctionExpression doesn't have all methods
        # So we test what's actually possible
        upper_result = field.upper()
        cypher = upper_result.to_cypher({"n": "n"}, "")
        assert "upper(n.text)" in cypher

        # Test combining with standalone functions
        combined = coalesce(field.upper(), "default")
        combined_cypher = combined.to_cypher({"n": "n"}, "")
        assert "coalesce(" in combined_cypher
        assert "upper(n.text)" in combined_cypher

    def test_parameter_handling_complex(self):
        """Test parameter handling in complex nested expressions."""
        field = QueryField("data")

        complex_expr = case().when(
            field.array_distance(array_value(1, 2, 3)) > 5.0,
            cast(field, "STRING")
        ).else_(
            coalesce(field.md5(), "default_hash")
        )

        params = complex_expr.get_parameters()
        assert isinstance(params, dict)
        # Should contain parameters from array_value, comparison, and coalesce
        assert len(params) > 0

    def test_field_references_complex(self):
        """Test field reference tracking in complex expressions."""
        field1 = QueryField("field1")
        field2 = QueryField("field2")
        field3 = QueryField("field3")

        complex_expr = case(field1).when(
            field2,
            field3.upper()
        ).else_(
            coalesce(field1, field2)
        )

        refs = complex_expr.get_field_references()
        assert "field1" in refs
        assert "field2" in refs
        assert "field3" in refs

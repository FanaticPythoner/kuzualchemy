# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
Tests for all Kuzu functions implemented in KuzuAlchemy.
Tests both QueryField methods and standalone functions.
"""

import pytest

from kuzualchemy import (
    KuzuBaseModel, kuzu_node, kuzu_field, KuzuDataType, Query,
    # Standalone functions
    pi, abs, sqrt, sin, cos, tan, concat, current_date, current_timestamp, make_date, date_part,
    to_years, to_months, to_days, to_hours, to_minutes, to_seconds,
    array_value, array_distance, array_dot_product, array_cosine_similarity,
    coalesce, ifnull, nullif, typeof, constant_or_null, count_if, error,
    md5, sha256, hash, gen_random_uuid, uuid,
    blob, encode, decode, octet_length,
    struct_extract, map_func, map_extract, element_at, cardinality,
    union_value, union_tag, union_extract,
    id_func, label, labels, offset,
    nodes, rels, properties, is_trail, is_acyclic, length, cost,
    cast, cast_as, case,
    list_creation, list_concat, range,
)


@kuzu_node("TestUser")
class TestUser(KuzuBaseModel):
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
    age: int = kuzu_field(kuzu_type=KuzuDataType.INT32)
    score: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE)


class TestNumericFunctions:
    """Test numeric functions - both QueryField methods and standalone functions."""
    
    def test_numeric_functions_queryfield(self):
        """Test numeric functions on QueryField objects."""
        query = Query(TestUser)
        
        # Test basic numeric functions
        abs_expr = query.fields.score.abs()
        assert abs_expr.function_name == "abs"
        assert abs_expr.args == [query.fields.score]
        
        sqrt_expr = query.fields.score.sqrt()
        assert sqrt_expr.function_name == "sqrt"
        
        sin_expr = query.fields.score.sin()
        assert sin_expr.function_name == "sin"
        
        # Test power function
        pow_expr = query.fields.score.pow(2)
        assert pow_expr.function_name == "pow"
        assert len(pow_expr.args) == 2
    
    def test_numeric_functions_standalone(self):
        """Test standalone numeric functions."""
        # Test pi function
        pi_expr = pi()
        assert pi_expr.function_name == "pi"
        assert pi_expr.args == []
        
        # Test functions with arguments
        abs_expr = abs(42)
        assert abs_expr.function_name == "abs"
        assert abs_expr.args == [42]
        
        sqrt_expr = sqrt(16)
        assert sqrt_expr.function_name == "sqrt"
        assert sqrt_expr.args == [16]
        
        # Test trigonometric functions
        sin_expr = sin(3.14159)
        assert sin_expr.function_name == "sin"
        
        cos_expr = cos(0)
        assert cos_expr.function_name == "cos"
        
        tan_expr = tan(0.785)
        assert tan_expr.function_name == "tan"
    
    def test_arithmetic_operators(self):
        """Test arithmetic operators on QueryField objects."""
        query = Query(TestUser)
        
        # Test addition
        add_expr = query.fields.age + 5
        cypher = add_expr.to_cypher({"n": "n"})
        assert "+" in cypher
        assert "n.age" in cypher

        # Test subtraction
        sub_expr = query.fields.age - 2
        cypher = sub_expr.to_cypher({"n": "n"})
        assert "-" in cypher

        # Test multiplication
        mul_expr = query.fields.score * 1.5
        cypher = mul_expr.to_cypher({"n": "n"})
        assert "*" in cypher

        # Test division
        div_expr = query.fields.score / 2
        cypher = div_expr.to_cypher({"n": "n"})
        assert "/" in cypher

        # Test modulo
        mod_expr = query.fields.age % 10
        cypher = mod_expr.to_cypher({"n": "n"})
        assert "%" in cypher

        # Test power
        pow_expr = query.fields.age ** 2
        cypher = pow_expr.to_cypher({"n": "n"})
        assert "^" in cypher


class TestTextFunctions:
    """Test text functions - both QueryField methods and standalone functions."""
    
    def test_text_functions_queryfield(self):
        """Test text functions on QueryField objects."""
        query = Query(TestUser)
        
        # Test string functions
        upper_expr = query.fields.name.upper()
        assert upper_expr.function_name == "upper"
        assert upper_expr.args == [query.fields.name]
        
        lower_expr = query.fields.name.lower()
        assert lower_expr.function_name == "lower"
        
        size_expr = query.fields.name.size()
        assert size_expr.function_name == "size"
        
        # Test substring
        substr_expr = query.fields.name.substring(1, 5)
        assert substr_expr.function_name == "substring"
        assert len(substr_expr.args) == 3
        
        # Test string operations
        concat_expr = query.fields.name.concat(" Smith")
        assert concat_expr.function_name == "concat"
        assert len(concat_expr.args) == 2
    
    def test_text_functions_standalone(self):
        """Test standalone text functions."""
        # Test concat function
        concat_expr = concat("Hello", " ", "World")
        assert concat_expr.function_name == "concat"
        assert concat_expr.args == ["Hello", " ", "World"]
        
        # Test string manipulation functions - these are QueryField methods, not standalone
        # We'll test them through QueryField objects instead
        pass
    
    def test_string_indexing_slicing(self):
        """Test string indexing and slicing operators."""
        query = Query(TestUser)
        
        # Test indexing
        index_expr = query.fields.name[0]
        assert index_expr.function_name == "array_extract"
        assert len(index_expr.args) == 2
        
        # Test slicing
        slice_expr = query.fields.name[1:5]
        assert slice_expr.function_name == "array_slice"
        assert len(slice_expr.args) == 3


class TestDateTimeFunctions:
    """Test date/time functions."""
    
    def test_datetime_functions_queryfield(self):
        """Test date/time functions on QueryField objects."""
        query = Query(TestUser)
        
        # Test date part extraction
        year_expr = query.fields.name.date_part("year")  # Assuming name field for test
        assert year_expr.function_name == "date_part"
        assert len(year_expr.args) == 2
        
        # Test date truncation
        trunc_expr = query.fields.name.date_trunc("month")
        assert trunc_expr.function_name == "date_trunc"
    
    def test_datetime_functions_standalone(self):
        """Test standalone date/time functions."""
        # Test current functions
        current_date_expr = current_date()
        assert current_date_expr.function_name == "current_date"
        assert current_date_expr.args == []
        
        current_ts_expr = current_timestamp()
        assert current_ts_expr.function_name == "current_timestamp"
        assert current_ts_expr.args == []
        
        # Test make_date
        make_date_expr = make_date(2023, 12, 25)
        assert make_date_expr.function_name == "make_date"
        assert make_date_expr.args == [2023, 12, 25]
        
        # Test date_part
        date_part_expr = date_part("year", "2023-12-25")
        assert date_part_expr.function_name == "date_part"
        assert date_part_expr.args == ["year", "2023-12-25"]


class TestIntervalFunctions:
    """Test interval functions."""
    
    def test_interval_functions_standalone(self):
        """Test standalone interval functions."""
        # Test interval creation functions
        years_expr = to_years(5)
        assert years_expr.function_name == "to_years"
        assert years_expr.args == [5]
        
        months_expr = to_months(3)
        assert months_expr.function_name == "to_months"
        
        days_expr = to_days(30)
        assert days_expr.function_name == "to_days"
        
        hours_expr = to_hours(24)
        assert hours_expr.function_name == "to_hours"
        
        minutes_expr = to_minutes(60)
        assert minutes_expr.function_name == "to_minutes"
        
        seconds_expr = to_seconds(3600)
        assert seconds_expr.function_name == "to_seconds"


class TestArrayFunctions:
    """Test array functions."""
    
    def test_array_functions_standalone(self):
        """Test standalone array functions."""
        # Test array creation
        array_expr = array_value(1, 2, 3, 4, 5)
        assert array_expr.function_name == "array_value"
        assert array_expr.args == [1, 2, 3, 4, 5]
        
        # Test array distance functions
        distance_expr = array_distance([1, 2, 3], [4, 5, 6])
        assert distance_expr.function_name == "array_distance"
        assert len(distance_expr.args) == 2
        
        dot_product_expr = array_dot_product([1, 2], [3, 4])
        assert dot_product_expr.function_name == "array_dot_product"
        
        cosine_sim_expr = array_cosine_similarity([1, 0], [0, 1])
        assert cosine_sim_expr.function_name == "array_cosine_similarity"


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_utility_functions_standalone(self):
        """Test standalone utility functions."""
        # Test coalesce
        coalesce_expr = coalesce(None, "default", "backup")
        assert coalesce_expr.function_name == "coalesce"
        assert coalesce_expr.args == [None, "default", "backup"]
        
        # Test ifnull
        ifnull_expr = ifnull("value", "default")
        assert ifnull_expr.function_name == "ifnull"
        assert len(ifnull_expr.args) == 2
        
        # Test nullif
        nullif_expr = nullif("a", "b")
        assert nullif_expr.function_name == "nullif"
        
        # Test typeof
        typeof_expr = typeof("hello")
        assert typeof_expr.function_name == "typeof"
        
        # Test additional utility functions
        const_null_expr = constant_or_null("const", "check")
        assert const_null_expr.function_name == "constant_or_null"
        
        count_if_expr = count_if(True)
        assert count_if_expr.function_name == "count_if"
        
        error_expr = error("Error message")
        assert error_expr.function_name == "error"


class TestHashFunctions:
    """Test hash functions."""
    
    def test_hash_functions_standalone(self):
        """Test standalone hash functions."""
        # Test MD5
        md5_expr = md5("hello")
        assert md5_expr.function_name == "md5"
        assert md5_expr.args == ["hello"]
        
        # Test SHA256
        sha256_expr = sha256("hello")
        assert sha256_expr.function_name == "sha256"
        
        # Test generic hash
        hash_expr = hash("hello")
        assert hash_expr.function_name == "hash"


class TestUUIDFunctions:
    """Test UUID functions."""
    
    def test_uuid_functions_standalone(self):
        """Test standalone UUID functions."""
        # Test random UUID generation
        random_uuid_expr = gen_random_uuid()
        assert random_uuid_expr.function_name == "gen_random_uuid"
        assert random_uuid_expr.args == []
        
        # Test UUID creation from string
        uuid_expr = uuid("550e8400-e29b-41d4-a716-446655440000")
        assert uuid_expr.function_name == "UUID"
        assert len(uuid_expr.args) == 1


class TestBlobFunctions:
    """Test blob functions."""

    def test_blob_functions_standalone(self):
        """Test standalone blob functions."""
        # Test blob creation
        blob_expr = blob("hello")
        assert blob_expr.function_name == "BLOB"
        assert blob_expr.args == ["hello"]

        # Test encode
        encode_expr = encode("hello")
        assert encode_expr.function_name == "encode"

        # Test decode
        decode_expr = decode("aGVsbG8=")
        assert decode_expr.function_name == "decode"

        # Test octet_length
        octet_expr = octet_length("hello")
        assert octet_expr.function_name == "octet_length"


class TestStructFunctions:
    """Test struct functions."""

    def test_struct_functions_standalone(self):
        """Test standalone struct functions."""
        # Test struct_extract
        extract_expr = struct_extract({"name": "John", "age": 30}, "name")
        assert extract_expr.function_name == "struct_extract"
        assert len(extract_expr.args) == 2

    def test_struct_operations_queryfield(self):
        """Test struct operations on QueryField objects."""
        query = Query(TestUser)

        # Test dot notation access (simulated)
        # This would be implemented as a special case in QueryField
        # For now, we test the underlying struct_extract functionality
        extract_expr = query.fields.name.struct_extract("property")
        assert extract_expr.function_name == "struct_extract"


class TestMapFunctions:
    """Test map functions."""

    def test_map_functions_standalone(self):
        """Test standalone map functions."""
        # Test map creation
        map_expr = map_func(["key1", "key2"], ["val1", "val2"])
        assert map_expr.function_name == "map"
        assert len(map_expr.args) == 2

        # Test map_extract
        extract_expr = map_extract({"key": "value"}, "key")
        assert extract_expr.function_name == "map_extract"

        # Test element_at
        element_expr = element_at({"a": 1, "b": 2}, "a")
        assert element_expr.function_name == "element_at"

        # Test cardinality
        card_expr = cardinality({"a": 1, "b": 2})
        assert card_expr.function_name == "cardinality"


class TestUnionFunctions:
    """Test union functions."""

    def test_union_functions_standalone(self):
        """Test standalone union functions."""
        # Test union_value
        union_expr = union_value("tag", "value")
        assert union_expr.function_name == "union_value"
        assert len(union_expr.args) == 1  # Combined as "tag := value"

        # Test union_tag
        tag_expr = union_tag("union_value")
        assert tag_expr.function_name == "union_tag"

        # Test union_extract
        extract_expr = union_extract("union_value", "tag")
        assert extract_expr.function_name == "union_extract"


class TestNodeRelFunctions:
    """Test node/relationship functions."""

    def test_node_rel_functions_standalone(self):
        """Test standalone node/relationship functions."""
        # Test ID function
        id_expr = id_func("node")
        assert id_expr.function_name == "ID"
        assert id_expr.args == ["node"]

        # Test LABEL function
        label_expr = label("node")
        assert label_expr.function_name == "LABEL"

        # Test LABELS function
        labels_expr = labels("node")
        assert labels_expr.function_name == "LABELS"

        # Test OFFSET function
        offset_expr = offset("rel")
        assert offset_expr.function_name == "OFFSET"


class TestRecursiveRelFunctions:
    """Test recursive relationship functions."""

    def test_recursive_rel_functions_standalone(self):
        """Test standalone recursive relationship functions."""
        # Test NODES function
        nodes_expr = nodes("path")
        assert nodes_expr.function_name == "NODES"
        assert nodes_expr.args == ["path"]

        # Test RELS function
        rels_expr = rels("path")
        assert rels_expr.function_name == "RELS"

        # Test PROPERTIES function
        props_expr = properties("path", "property_name")
        assert props_expr.function_name == "PROPERTIES"

        # Test IS_TRAIL function
        trail_expr = is_trail("path")
        assert trail_expr.function_name == "IS_TRAIL"

        # Test IS_ACYCLIC function
        acyclic_expr = is_acyclic("path")
        assert acyclic_expr.function_name == "IS_ACYCLIC"

        # Test LENGTH function
        length_expr = length("path")
        assert length_expr.function_name == "LENGTH"

        # Test COST function
        cost_expr = cost("path")
        assert cost_expr.function_name == "COST"


class TestCastingFunctions:
    """Test casting functions."""

    def test_casting_functions_standalone(self):
        """Test standalone casting functions."""
        # Test CAST function
        cast_expr = cast("123", "INT64")
        assert hasattr(cast_expr, 'value')
        assert hasattr(cast_expr, 'target_type')

        # Test CAST AS syntax
        cast_as_expr = cast_as("123", "INT64")
        assert hasattr(cast_as_expr, 'value')
        assert hasattr(cast_as_expr, 'target_type')

    def test_casting_queryfield(self):
        """Test casting on QueryField objects."""
        query = Query(TestUser)

        # Test cast method
        cast_expr = query.fields.age.cast("STRING")
        # CastExpression has different attributes than FunctionExpression
        assert hasattr(cast_expr, 'value')
        assert hasattr(cast_expr, 'target_type')


class TestCaseExpressions:
    """Test CASE expressions."""

    def test_case_expressions_standalone(self):
        """Test standalone CASE expressions."""
        # Test case function
        case_expr = case()
        assert hasattr(case_expr, 'when')
        assert hasattr(case_expr, 'else_')

    def test_case_expressions_queryfield(self):
        """Test CASE expressions on QueryField objects."""
        query = Query(TestUser)

        # Test case method
        case_expr = query.fields.age.case()
        assert hasattr(case_expr, 'when')
        assert hasattr(case_expr, 'else_')


class TestListFunctions:
    """Test list functions."""

    def test_list_functions_standalone(self):
        """Test standalone list functions."""
        # Test list_creation
        list_expr = list_creation(1, 2, 3, 4, 5)
        assert list_expr.function_name == "list_creation"
        assert list_expr.args == [1, 2, 3, 4, 5]

        # Test list_concat
        concat_expr = list_concat([1, 2], [3, 4])
        assert concat_expr.function_name == "list_concat"
        assert len(concat_expr.args) == 2

        # Test range
        range_expr = range(1, 10, 2)
        assert range_expr.function_name == "range"
        assert range_expr.args == [1, 10, 2]


class TestComparisonOperators:
    """Test comparison operators."""

    def test_comparison_operators_queryfield(self):
        """Test comparison operators on QueryField objects."""
        query = Query(TestUser)

        # Test equality
        eq_expr = query.fields.age == 25
        cypher = eq_expr.to_cypher({"n": "n"})
        assert "=" in cypher
        assert "n.age" in cypher

        # Test inequality
        ne_expr = query.fields.age != 25
        cypher = ne_expr.to_cypher({"n": "n"})
        assert "<>" in cypher

        # Test greater than
        gt_expr = query.fields.age > 18
        cypher = gt_expr.to_cypher({"n": "n"})
        assert ">" in cypher

        # Test less than
        lt_expr = query.fields.age < 65
        cypher = lt_expr.to_cypher({"n": "n"})
        assert "<" in cypher

        # Test greater than or equal
        gte_expr = query.fields.age >= 18
        cypher = gte_expr.to_cypher({"n": "n"})
        assert ">=" in cypher

        # Test less than or equal
        lte_expr = query.fields.age <= 65
        cypher = lte_expr.to_cypher({"n": "n"})
        assert "<=" in cypher


class TestLogicalOperators:
    """Test logical operators."""

    def test_logical_operators_queryfield(self):
        """Test logical operators on QueryField objects."""
        query = Query(TestUser)

        # Test AND
        and_expr = (query.fields.age > 18) & (query.fields.age < 65)
        cypher = and_expr.to_cypher({"n": "n"})
        assert "AND" in cypher

        # Test OR
        or_expr = (query.fields.age < 18) | (query.fields.age > 65)
        cypher = or_expr.to_cypher({"n": "n"})
        assert "OR" in cypher

        # Test NOT
        not_expr = ~(query.fields.age == 25)
        cypher = not_expr.to_cypher({"n": "n"})
        assert "NOT" in cypher


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

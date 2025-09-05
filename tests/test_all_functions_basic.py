# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
Basic test to verify all major function categories are working.
"""

import pytest
from kuzualchemy import (
    # Text functions
    concat, upper, lower, contains, substring, trim, lpad, pi, abs, ceil, floor, sqrt, sin, list_creation, list_concat, list_append, list_contains, list_slice, list_sort,
    # Date functions
    current_date, current_timestamp, make_date, date_part,
    # Array functions
    array_value, array_distance, array_dot_product,
    # Utility functions
    coalesce, ifnull, nullif, typeof,
    # Hash functions
    md5, sha256, hash,
    # UUID functions
    gen_random_uuid, uuid,
    # Blob functions
    blob, encode, decode, octet_length,
    # Cast and case
    cast, cast_as, case,
    # Interval functions
    to_years, to_months, to_days,
)


class TestAllFunctionCategories:
    """Test that all major function categories are implemented and working."""
    
    def test_text_functions(self):
        """Test text functions work correctly."""
        # Test basic text functions
        concat_expr = concat("Hello", " ", "World")
        assert concat_expr.function_name == "concat"
        assert concat_expr.args == ["Hello", " ", "World"]
        
        upper_expr = upper("hello")
        assert upper_expr.function_name == "upper"
        assert upper_expr.args == ["hello"]
        
        lower_expr = lower("HELLO")
        assert lower_expr.function_name == "lower"
        
        contains_expr = contains("hello world", "world")
        assert contains_expr.function_name == "contains"
        
        substring_expr = substring("hello", 1, 3)
        assert substring_expr.function_name == "substring"
        assert len(substring_expr.args) == 3
        
        trim_expr = trim("  hello  ")
        assert trim_expr.function_name == "trim"
        
        lpad_expr = lpad("hello", 10, "*")
        assert lpad_expr.function_name == "lpad"
        assert len(lpad_expr.args) == 3
    
    def test_numeric_functions(self):
        """Test numeric functions work correctly."""
        pi_expr = pi()
        assert pi_expr.function_name == "pi"
        assert pi_expr.args == []
        
        abs_expr = abs(-5)
        assert abs_expr.function_name == "abs"
        assert abs_expr.args == [-5]
        
        sqrt_expr = sqrt(16)
        assert sqrt_expr.function_name == "sqrt"
        
        sin_expr = sin(3.14159)
        assert sin_expr.function_name == "sin"
        
        ceil_expr = ceil(4.2)
        assert ceil_expr.function_name == "ceil"
        
        floor_expr = floor(4.8)
        assert floor_expr.function_name == "floor"
    
    def test_list_functions(self):
        """Test list functions work correctly."""
        list_expr = list_creation(1, 2, 3, 4, 5)
        assert list_expr.function_name == "list_creation"
        assert list_expr.args == [1, 2, 3, 4, 5]
        
        concat_expr = list_concat([1, 2], [3, 4])
        assert concat_expr.function_name == "list_concat"
        assert len(concat_expr.args) == 2
        
        append_expr = list_append([1, 2], 3)
        assert append_expr.function_name == "list_append"
        
        contains_expr = list_contains([1, 2, 3], 2)
        assert contains_expr.function_name == "list_contains"
        
        slice_expr = list_slice([1, 2, 3, 4], 1, 3)
        assert slice_expr.function_name == "list_slice"
        
        sort_expr = list_sort([3, 1, 2])
        assert sort_expr.function_name == "list_sort"
    
    def test_date_functions(self):
        """Test date functions work correctly."""
        current_date_expr = current_date()
        assert current_date_expr.function_name == "current_date"
        assert current_date_expr.args == []
        
        current_ts_expr = current_timestamp()
        assert current_ts_expr.function_name == "current_timestamp"
        
        make_date_expr = make_date(2023, 12, 25)
        assert make_date_expr.function_name == "make_date"
        assert make_date_expr.args == [2023, 12, 25]
        
        date_part_expr = date_part("year", "2023-12-25")
        assert date_part_expr.function_name == "date_part"
        assert len(date_part_expr.args) == 2
    
    def test_array_functions(self):
        """Test array functions work correctly."""
        array_expr = array_value(1, 2, 3, 4)
        assert array_expr.function_name == "array_value"
        assert array_expr.args == [1, 2, 3, 4]
        
        distance_expr = array_distance([1, 2], [3, 4])
        assert distance_expr.function_name == "array_distance"
        assert len(distance_expr.args) == 2
        
        dot_product_expr = array_dot_product([1, 2], [3, 4])
        assert dot_product_expr.function_name == "array_dot_product"
    
    def test_utility_functions(self):
        """Test utility functions work correctly."""
        coalesce_expr = coalesce(None, "default", "backup")
        assert coalesce_expr.function_name == "coalesce"
        assert coalesce_expr.args == [None, "default", "backup"]
        
        ifnull_expr = ifnull("value", "default")
        assert ifnull_expr.function_name == "ifnull"
        assert len(ifnull_expr.args) == 2
        
        nullif_expr = nullif("a", "b")
        assert nullif_expr.function_name == "nullif"
        
        typeof_expr = typeof("hello")
        assert typeof_expr.function_name == "typeof"
    
    def test_hash_functions(self):
        """Test hash functions work correctly."""
        md5_expr = md5("hello")
        assert md5_expr.function_name == "md5"
        assert md5_expr.args == ["hello"]
        
        sha256_expr = sha256("hello")
        assert sha256_expr.function_name == "sha256"
        
        hash_expr = hash("hello")
        assert hash_expr.function_name == "hash"
    
    def test_uuid_functions(self):
        """Test UUID functions work correctly."""
        random_uuid_expr = gen_random_uuid()
        assert random_uuid_expr.function_name == "gen_random_uuid"
        assert random_uuid_expr.args == []
        
        uuid_expr = uuid("550e8400-e29b-41d4-a716-446655440000")
        assert uuid_expr.function_name == "UUID"
        assert len(uuid_expr.args) == 1
    
    def test_blob_functions(self):
        """Test blob functions work correctly."""
        blob_expr = blob("hello")
        assert blob_expr.function_name == "BLOB"
        assert blob_expr.args == ["hello"]
        
        encode_expr = encode("hello")
        assert encode_expr.function_name == "encode"
        
        decode_expr = decode("aGVsbG8=")
        assert decode_expr.function_name == "decode"
        
        octet_expr = octet_length("hello")
        assert octet_expr.function_name == "octet_length"
    
    def test_cast_and_case_functions(self):
        """Test cast and case functions work correctly."""
        cast_expr = cast("123", "INT64")
        assert hasattr(cast_expr, 'value')
        assert hasattr(cast_expr, 'target_type')
        assert cast_expr.value == "123"
        assert cast_expr.target_type == "INT64"

        cast_as_expr = cast_as("123", "INT64")
        assert hasattr(cast_as_expr, 'value')
        assert hasattr(cast_as_expr, 'target_type')
        assert cast_as_expr.use_as_syntax == True

        case_expr = case()
        assert hasattr(case_expr, 'when')
        assert hasattr(case_expr, 'else_')
    
    def test_interval_functions(self):
        """Test interval functions work correctly."""
        years_expr = to_years(5)
        assert years_expr.function_name == "to_years"
        assert years_expr.args == [5]
        
        months_expr = to_months(3)
        assert months_expr.function_name == "to_months"
        
        days_expr = to_days(30)
        assert days_expr.function_name == "to_days"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

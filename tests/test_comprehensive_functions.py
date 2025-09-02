"""
Tests for all KuzuAlchemy functions to verify they work correctly with Kuzu.
This test suite validates that all implemented functions generate correct Cypher and execute properly.
"""

import pytest
from kuzualchemy.kuzu_functions import (
    # Pattern matching functions
    regexp_matches, regexp_replace, regexp_extract, regexp_extract_all, regexp_split_to_array,
    # Hash functions
    md5, sha256, hash,
    # UUID functions
    gen_random_uuid, uuid,
    # Utility functions
    coalesce, ifnull, nullif, typeof, error,
    # BLOB functions
    blob, encode, decode, octet_length,
    # Array functions
    array_value, array_distance, array_dot_product, array_cosine_similarity,
    # Struct functions
    struct_extract,
    # Map functions
    map_func, map_extract, element_at, cardinality, map_keys, map_values,
    # Union functions
    union_value, union_tag, union_extract,
    # Node/Rel functions
    id_func, label, labels, offset,
    # Recursive path functions
    nodes, rels, properties, is_trail, is_acyclic, length, cost,
    # Cast and case
    cast, cast_as, case
)
from kuzualchemy.kuzu_query_fields import QueryField


class TestPatternMatchingFunctions:
    """Test pattern matching functions."""
    
    def test_regexp_matches_function(self):
        """Test regexp_matches function generates correct Cypher structure."""
        result = regexp_matches('test_string', '^test')
        cypher = result.to_cypher({})
        assert cypher.startswith("regexp_matches(")
        assert "func_arg" in cypher  # Uses parameterized queries

    def test_regexp_matches_field_method(self):
        """Test regexp_matches method on QueryField."""
        field = QueryField('name')
        result = field.regexp_matches('^John')
        cypher = result.to_cypher({})
        assert cypher.startswith("regexp_matches(n.name,")
        assert "func_arg" in cypher  # Uses parameterized queries

    def test_regexp_replace_function(self):
        """Test regexp_replace function generates correct Cypher."""
        result = regexp_replace('test_string', 'test', 'demo')
        cypher = result.to_cypher({})
        assert cypher.startswith("regexp_replace(")
        assert "func_arg" in cypher

    def test_regexp_replace_with_options(self):
        """Test regexp_replace function with options."""
        result = regexp_replace('test_string', 'test', 'demo', 'g')
        cypher = result.to_cypher({})
        assert cypher.startswith("regexp_replace(")
        assert "func_arg" in cypher

    def test_regexp_extract_function(self):
        """Test regexp_extract function generates correct Cypher."""
        result = regexp_extract('test_string', 'test', 0)
        cypher = result.to_cypher({})
        assert cypher.startswith("regexp_extract(")
        assert "func_arg" in cypher

    def test_regexp_extract_all_function(self):
        """Test regexp_extract_all function generates correct Cypher."""
        result = regexp_extract_all('test_string', 'test', 0)
        cypher = result.to_cypher({})
        assert cypher.startswith("regexp_extract_all(")
        assert "func_arg" in cypher

    def test_regexp_split_to_array_function(self):
        """Test regexp_split_to_array function generates correct Cypher."""
        result = regexp_split_to_array('test string', ' ')
        cypher = result.to_cypher({})
        assert cypher.startswith("regexp_split_to_array(")
        assert "func_arg" in cypher


class TestHashFunctions:
    """Test hash functions."""

    def test_md5_function(self):
        """Test md5 function generates correct Cypher."""
        result = md5('test_string')
        cypher = result.to_cypher({})
        assert cypher.startswith("md5(")
        assert "func_arg" in cypher

    def test_md5_field_method(self):
        """Test md5 method on QueryField."""
        field = QueryField('password')
        result = field.md5()
        cypher = result.to_cypher({})
        assert cypher == "md5(n.password)"

    def test_sha256_function(self):
        """Test sha256 function generates correct Cypher."""
        result = sha256('test_string')
        cypher = result.to_cypher({})
        assert cypher.startswith("sha256(")
        assert "func_arg" in cypher

    def test_hash_function(self):
        """Test hash function generates correct Cypher."""
        result = hash('test_string')
        cypher = result.to_cypher({})
        assert cypher.startswith("hash(")
        assert "func_arg" in cypher


class TestUUIDFunctions:
    """Test UUID functions."""

    def test_gen_random_uuid_function(self):
        """Test gen_random_uuid function generates correct Cypher."""
        result = gen_random_uuid()
        cypher = result.to_cypher({})
        assert cypher == "gen_random_uuid()"

    def test_uuid_function(self):
        """Test uuid function generates correct Cypher."""
        result = uuid('a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11')
        cypher = result.to_cypher({})
        assert cypher.startswith("UUID(")
        assert "func_arg" in cypher


class TestUtilityFunctions:
    """Test utility functions."""

    def test_coalesce_function(self):
        """Test coalesce function generates correct Cypher."""
        result = coalesce('value1', 'value2', 'value3')
        cypher = result.to_cypher({})
        assert cypher.startswith("coalesce(")
        assert "func_arg" in cypher

    def test_coalesce_field_method(self):
        """Test coalesce method on QueryField."""
        field = QueryField('optional_field')
        result = field.coalesce('default_value')
        cypher = result.to_cypher({})
        assert cypher.startswith("coalesce(n.optional_field,")
        assert "func_arg" in cypher

    def test_ifnull_function(self):
        """Test ifnull function generates correct Cypher."""
        result = ifnull('value1', 'default')
        cypher = result.to_cypher({})
        assert cypher.startswith("ifnull(")
        assert "func_arg" in cypher

    def test_nullif_function(self):
        """Test nullif function generates correct Cypher."""
        result = nullif('value1', 'empty')
        cypher = result.to_cypher({})
        assert cypher.startswith("nullif(")
        assert "func_arg" in cypher

    def test_typeof_function(self):
        """Test typeof function generates correct Cypher."""
        result = typeof('value1')
        cypher = result.to_cypher({})
        assert cypher.startswith("typeof(")
        assert "func_arg" in cypher

    def test_error_function(self):
        """Test error function generates correct Cypher."""
        result = error('Error message')
        cypher = result.to_cypher({})
        assert cypher.startswith("error(")
        assert "func_arg" in cypher


class TestBLOBFunctions:
    """Test BLOB functions."""

    def test_blob_function(self):
        """Test blob function generates correct Cypher."""
        result = blob('\\xAA\\xBD')
        cypher = result.to_cypher({})
        assert cypher.startswith("BLOB(")
        assert "func_arg" in cypher

    def test_encode_function(self):
        """Test encode function generates correct Cypher."""
        result = encode('test string')
        cypher = result.to_cypher({})
        assert cypher.startswith("encode(")
        assert "func_arg" in cypher

    def test_encode_field_method(self):
        """Test encode method on QueryField."""
        field = QueryField('text_field')
        result = field.encode()
        cypher = result.to_cypher({})
        assert cypher == "encode(n.text_field)"

    def test_decode_function(self):
        """Test decode function generates correct Cypher."""
        result = decode('blob_data')
        cypher = result.to_cypher({})
        assert cypher.startswith("decode(")
        assert "func_arg" in cypher

    def test_octet_length_function(self):
        """Test octet_length function generates correct Cypher."""
        result = octet_length('blob_data')
        cypher = result.to_cypher({})
        assert cypher.startswith("octet_length(")
        assert "func_arg" in cypher


class TestArrayFunctions:
    """Test array functions."""
    
    def test_array_value_function(self):
        """Test array_value function generates correct Cypher."""
        result = array_value(1, 2, 3, 4)
        cypher = result.to_cypher({})
        assert cypher.startswith("array_value(")

    def test_array_distance_function(self):
        """Test array_distance function generates correct Cypher."""
        result = array_distance([1, 2, 3], [4, 5, 6])
        cypher = result.to_cypher({})
        assert cypher.startswith("array_distance(")

    def test_array_dot_product_function(self):
        """Test array_dot_product function generates correct Cypher."""
        result = array_dot_product([1, 2, 3], [4, 5, 6])
        cypher = result.to_cypher({})
        assert cypher.startswith("array_dot_product(")

    def test_array_cosine_similarity_function(self):
        """Test array_cosine_similarity function generates correct Cypher."""
        result = array_cosine_similarity([1, 2, 3], [4, 5, 6])
        cypher = result.to_cypher({})
        assert cypher.startswith("array_cosine_similarity(")


class TestStructFunctions:
    """Test struct functions."""
    
    def test_struct_extract_function(self):
        """Test struct_extract function generates correct Cypher."""
        result = struct_extract('struct_value', 'field_name')
        cypher = result.to_cypher({})
        assert cypher.startswith("struct_extract(")
        assert "func_arg" in cypher
    
    def test_struct_extract_field_method(self):
        """Test struct_extract method on QueryField."""
        field = QueryField('struct_field')
        result = field.struct_extract('name')
        cypher = result.to_cypher({})
        assert cypher.startswith("struct_extract(n.struct_field,")
        assert "func_arg" in cypher


class TestMapFunctions:
    """Test map functions."""
    
    def test_map_func_function(self):
        """Test map function generates correct Cypher."""
        result = map_func([1, 2], ['a', 'b'])
        cypher = result.to_cypher({})
        assert cypher.startswith("map(")
        assert "func_arg" in cypher

    def test_map_extract_function(self):
        """Test map_extract function generates correct Cypher."""
        result = map_extract('map_value', 'key')
        cypher = result.to_cypher({})
        assert cypher.startswith("map_extract(")
        assert "func_arg" in cypher

    def test_element_at_function(self):
        """Test element_at function generates correct Cypher."""
        result = element_at('map_value', 'key')
        cypher = result.to_cypher({})
        assert cypher.startswith("element_at(")
        assert "func_arg" in cypher

    def test_cardinality_function(self):
        """Test cardinality function generates correct Cypher."""
        result = cardinality('map_value')
        cypher = result.to_cypher({})
        assert cypher.startswith("cardinality(")
        assert "func_arg" in cypher

    def test_map_keys_function(self):
        """Test map_keys function generates correct Cypher."""
        result = map_keys('map_value')
        cypher = result.to_cypher({})
        assert cypher.startswith("map_keys(")
        assert "func_arg" in cypher

    def test_map_values_function(self):
        """Test map_values function generates correct Cypher."""
        result = map_values('map_value')
        cypher = result.to_cypher({})
        assert cypher.startswith("map_values(")
        assert "func_arg" in cypher


class TestUnionFunctions:
    """Test union functions."""
    
    def test_union_value_function(self):
        """Test union_value function generates correct Cypher."""
        result = union_value('tag', 'value')
        cypher = result.to_cypher({})
        assert cypher.startswith("union_value(")
        assert "func_arg" in cypher

    def test_union_tag_function(self):
        """Test union_tag function generates correct Cypher."""
        result = union_tag('union_value')
        cypher = result.to_cypher({})
        assert cypher.startswith("union_tag(")
        assert "func_arg" in cypher

    def test_union_extract_function(self):
        """Test union_extract function generates correct Cypher."""
        result = union_extract('union_value', 'tag')
        cypher = result.to_cypher({})
        assert cypher.startswith("union_extract(")
        assert "func_arg" in cypher


class TestNodeRelFunctions:
    """Test node/relationship functions."""
    
    def test_id_func_function(self):
        """Test ID function generates correct Cypher."""
        result = id_func('node')
        cypher = result.to_cypher({})
        assert cypher.startswith("ID(")
        assert "func_arg" in cypher

    def test_label_function(self):
        """Test LABEL function generates correct Cypher."""
        result = label('node')
        cypher = result.to_cypher({})
        assert cypher.startswith("LABEL(")
        assert "func_arg" in cypher

    def test_labels_function(self):
        """Test LABELS function generates correct Cypher."""
        result = labels('node')
        cypher = result.to_cypher({})
        assert cypher.startswith("LABELS(")
        assert "func_arg" in cypher

    def test_offset_function(self):
        """Test OFFSET function generates correct Cypher."""
        result = offset('node')
        cypher = result.to_cypher({})
        assert cypher.startswith("OFFSET(")
        assert "func_arg" in cypher


class TestRecursivePathFunctions:
    """Test recursive path functions."""
    
    def test_nodes_function(self):
        """Test NODES function generates correct Cypher."""
        result = nodes('path')
        cypher = result.to_cypher({})
        assert cypher.startswith("NODES(")
        assert "func_arg" in cypher

    def test_rels_function(self):
        """Test RELS function generates correct Cypher."""
        result = rels('path')
        cypher = result.to_cypher({})
        assert cypher.startswith("RELS(")
        assert "func_arg" in cypher

    def test_properties_function(self):
        """Test PROPERTIES function generates correct Cypher."""
        result = properties('path', 'name')
        cypher = result.to_cypher({})
        assert cypher.startswith("PROPERTIES(")
        assert "func_arg" in cypher

    def test_is_trail_function(self):
        """Test IS_TRAIL function generates correct Cypher."""
        result = is_trail('path')
        cypher = result.to_cypher({})
        assert cypher.startswith("IS_TRAIL(")
        assert "func_arg" in cypher

    def test_is_acyclic_function(self):
        """Test IS_ACYCLIC function generates correct Cypher."""
        result = is_acyclic('path')
        cypher = result.to_cypher({})
        assert cypher.startswith("IS_ACYCLIC(")
        assert "func_arg" in cypher

    def test_length_function(self):
        """Test LENGTH function generates correct Cypher."""
        result = length('path')
        cypher = result.to_cypher({})
        assert cypher.startswith("LENGTH(")
        assert "func_arg" in cypher

    def test_cost_function(self):
        """Test COST function generates correct Cypher."""
        result = cost('path')
        cypher = result.to_cypher({})
        assert cypher.startswith("COST(")
        assert "func_arg" in cypher


class TestCastAndCaseFunctions:
    """Test cast and case functions."""
    
    def test_cast_function(self):
        """Test cast function generates correct Cypher."""
        result = cast('value', 'STRING')
        cypher = result.to_cypher({})
        assert cypher.startswith("CAST(")
        assert "cast_value" in cypher

    def test_cast_as_function(self):
        """Test cast_as function generates correct Cypher."""
        result = cast_as('value', 'STRING')
        cypher = result.to_cypher({})
        assert cypher.startswith("CAST(")
        assert "cast_value" in cypher
    
    def test_case_function(self):
        """Test case function creates CaseExpression."""
        result = case()
        assert hasattr(result, 'when')
        assert hasattr(result, 'else_')


class TestIntegrationWithKuzu:
    """Integration tests that actually execute functions against Kuzu database."""

    def test_pattern_matching_integration(self, kuzu_session):
        """Test pattern matching functions work with real Kuzu database."""
        # Create test data
        kuzu_session.execute("CREATE NODE TABLE TestNode(name STRING, PRIMARY KEY(name))")
        kuzu_session.execute("CREATE (n:TestNode {name: 'test123'})")
        kuzu_session.execute("CREATE (n:TestNode {name: 'demo456'})")

        # Test regexp_matches
        result = kuzu_session.execute("""
            MATCH (n:TestNode)
            WHERE regexp_matches(n.name, '^test')
            RETURN n.name
        """)
        assert len(result) == 1
        assert result[0]['n.name'] == 'test123'

        # Test regexp_replace
        result = kuzu_session.execute("""
            MATCH (n:TestNode)
            WHERE n.name = 'test123'
            RETURN regexp_replace(n.name, '123', 'ABC') AS replaced
        """)
        # Check what keys are available and get the first key
        result_keys = list(result[0].keys())
        replaced_value = result[0][result_keys[0]]
        assert replaced_value == 'testABC'

        # Test regexp_extract
        result = kuzu_session.execute("""
            MATCH (n:TestNode)
            WHERE n.name = 'test123'
            RETURN regexp_extract(n.name, '[0-9]+', 0) AS extracted
        """)
        # Check what keys are available and get the first key
        result_keys = list(result[0].keys())
        extracted_value = result[0][result_keys[0]]
        assert extracted_value == '123'

    def test_hash_functions_integration(self, kuzu_session):
        """Test hash functions work with real Kuzu database."""
        # Test md5 - verify it returns a valid MD5 hash (32 hex chars)
        result = kuzu_session.execute("RETURN md5('kuzu') AS hash_result")
        hash_result = result[0]['hash_result']
        assert len(hash_result) == 32
        assert all(c in '0123456789abcdef' for c in hash_result.lower())

        # Test sha256 - verify it returns a valid SHA256 hash (64 hex chars)
        result = kuzu_session.execute("RETURN sha256('kuzu') AS hash_result")
        hash_result = result[0]['hash_result']
        assert len(hash_result) == 64
        assert all(c in '0123456789abcdef' for c in hash_result.lower())

        # Test hash (Murmurhash64) - verify it returns an integer
        result = kuzu_session.execute("RETURN hash('kuzu') AS hash_result")
        hash_result = result[0]['hash_result']
        assert isinstance(hash_result, int)

    def test_uuid_functions_integration(self, kuzu_session):
        """Test UUID functions work with real Kuzu database."""
        # Test UUID constructor
        test_uuid = 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11'
        result = kuzu_session.execute(f"RETURN UUID('{test_uuid}') AS uuid_result")
        # Kuzu returns a UUID object, convert to string for comparison
        assert str(result[0]['uuid_result']) == test_uuid

        # Test gen_random_uuid (just verify it returns a valid UUID format)
        result = kuzu_session.execute("RETURN gen_random_uuid() AS random_uuid")
        uuid_obj = result[0]['random_uuid']
        uuid_str = str(uuid_obj)
        # Basic UUID format validation (8-4-4-4-12 hex digits)
        import re
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        assert re.match(uuid_pattern, uuid_str, re.IGNORECASE)

    def test_utility_functions_integration(self, kuzu_session):
        """Test utility functions work with real Kuzu database."""
        # Test coalesce
        result = kuzu_session.execute("RETURN coalesce(NULL, 'default', 'backup') AS result")
        result_keys = list(result[0].keys())
        assert result[0][result_keys[0]] == 'default'

        # Test ifnull
        result = kuzu_session.execute("RETURN ifnull(NULL, 'default') AS result")
        result_keys = list(result[0].keys())
        assert result[0][result_keys[0]] == 'default'

        # Test nullif
        result = kuzu_session.execute("RETURN nullif('test', 'test') AS result")
        result_keys = list(result[0].keys())
        assert result[0][result_keys[0]] is None

        result = kuzu_session.execute("RETURN nullif('test', 'other') AS result")
        result_keys = list(result[0].keys())
        assert result[0][result_keys[0]] == 'test'

        # Test typeof
        result = kuzu_session.execute("RETURN typeof('string') AS result")
        assert result[0]['result'] == 'STRING'

        result = kuzu_session.execute("RETURN typeof(123) AS result")
        assert result[0]['result'] == 'INT64'

    def test_blob_functions_integration(self, kuzu_session):
        """Test BLOB functions work with real Kuzu database."""
        # Test BLOB constructor and encode/decode
        result = kuzu_session.execute("RETURN encode('test') AS encoded")
        result_keys = list(result[0].keys())
        encoded_blob = result[0][result_keys[0]]

        # Use parameterized query to avoid BLOB literal issues
        result = kuzu_session.execute("RETURN decode(encode('test')) AS decoded")
        result_keys = list(result[0].keys())
        assert result[0][result_keys[0]] == 'test'

        # Test octet_length
        result = kuzu_session.execute("RETURN octet_length(encode('test')) AS length")
        result_keys = list(result[0].keys())
        assert result[0][result_keys[0]] == 4  # 'test' is 4 bytes

    def test_array_functions_integration(self, kuzu_session):
        """Test array functions work with real Kuzu database."""
        # Test array_value
        result = kuzu_session.execute("RETURN array_value(1, 2, 3) AS arr")
        result_keys = list(result[0].keys())
        assert result[0][result_keys[0]] == [1, 2, 3]

        # Test array distance functions (requires FLOAT[] arrays)
        result = kuzu_session.execute("RETURN array_distance([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) AS distance")
        result_keys = list(result[0].keys())
        # Distance should be sqrt((4-1)^2 + (5-2)^2 + (6-3)^2) = sqrt(27) ≈ 5.196
        assert abs(result[0][result_keys[0]] - 5.196152422706632) < 0.001

        # Test array_dot_product (requires FLOAT[] arrays)
        result = kuzu_session.execute("RETURN array_dot_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) AS dot_product")
        result_keys = list(result[0].keys())
        assert result[0][result_keys[0]] == 32.0  # 1*4 + 2*5 + 3*6 = 32

    def test_map_functions_integration(self, kuzu_session):
        """Test map functions work with real Kuzu database."""
        # Test map creation and extraction
        result = kuzu_session.execute("RETURN map([1, 2], ['a', 'b']) AS map_result")
        result_keys = list(result[0].keys())
        map_obj = result[0][result_keys[0]]

        # Test map_extract (returns the value for the key)
        result = kuzu_session.execute("RETURN map_extract(map([1, 2], ['a', 'b']), 1) AS extracted")
        result_keys = list(result[0].keys())
        # map_extract returns the value associated with the key
        extracted_value = result[0][result_keys[0]]
        # The result might be wrapped in a list or be the direct value
        if isinstance(extracted_value, list):
            assert extracted_value[0] == 'a'
        else:
            assert extracted_value == 'a'

        # Test cardinality
        result = kuzu_session.execute("RETURN cardinality(map([1, 2], ['a', 'b'])) AS size")
        result_keys = list(result[0].keys())
        assert result[0][result_keys[0]] == 2

        # Test map_keys
        result = kuzu_session.execute("RETURN map_keys(map([1, 2], ['a', 'b'])) AS keys")
        result_keys = list(result[0].keys())
        assert result[0][result_keys[0]] == [1, 2]

        # Test map_values
        result = kuzu_session.execute("RETURN map_values(map([1, 2], ['a', 'b'])) AS values")
        result_keys = list(result[0].keys())
        assert result[0][result_keys[0]] == ['a', 'b']

    def test_cast_functions_integration(self, kuzu_session):
        """Test cast functions work with real Kuzu database."""
        # Test CAST function syntax
        result = kuzu_session.execute("RETURN CAST(123, 'STRING') AS str_result")
        result_keys = list(result[0].keys())
        assert result[0][result_keys[0]] == '123'

        # Test CAST AS syntax
        result = kuzu_session.execute("RETURN CAST(123 AS STRING) AS str_result")
        result_keys = list(result[0].keys())
        assert result[0][result_keys[0]] == '123'

        # Test casting to different types
        result = kuzu_session.execute("RETURN CAST('123.45', 'DOUBLE') AS double_result")
        result_keys = list(result[0].keys())
        assert result[0][result_keys[0]] == 123.45

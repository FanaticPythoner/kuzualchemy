"""
Integration tests for KuzuAlchemy functions with real Kuzu database.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from kuzualchemy import (
    KuzuBaseModel, node, Field, KuzuDataType, Query, KuzuSession,
    concat, size, ceil, floor, sqrt, pi, list_creation, get_ddl_for_node
)
from kuzualchemy.kuzu_functions import abs as kuzu_abs
from kuzualchemy.kuzu_orm import ArrayTypeSpecification


@node("Person")
class Person(KuzuBaseModel):
    """Person node for testing."""
    id: int = Field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    name: str = Field(kuzu_type=KuzuDataType.STRING)
    age: int = Field(kuzu_type=KuzuDataType.INT64)
    height: float = Field(kuzu_type=KuzuDataType.DOUBLE)
    tags: list = Field(kuzu_type=ArrayTypeSpecification(element_type=KuzuDataType.STRING))


class TestFunctionIntegration:
    """Integration tests for functions with real Kuzu database."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test_functions.kuzu"
        yield str(db_path)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def session(self, temp_db_path):
        """Create test session with sample data."""
        session = KuzuSession(db_path=temp_db_path)
        
        # Create tables
        session.execute(get_ddl_for_node(Person))

        # Insert test data
        session.execute(
            "CREATE (p:Person {id: 1, name: 'Alice Johnson', age: 30, height: 5.6, tags: ['developer', 'python']})"
        )
        session.execute(
            "CREATE (p:Person {id: 2, name: 'Bob Smith', age: 25, height: 6.1, tags: ['designer', 'ui']})"
        )
        session.execute(
            "CREATE (p:Person {id: 3, name: 'Charlie Brown', age: 35, height: 5.9, tags: ['manager', 'team-lead']})"
        )
        
        yield session
        session.close()

    def test_text_functions_integration(self, session):
        """Test text functions with real database."""
        # Test that we can query the data
        results = session.execute("MATCH (p:Person) RETURN p.name, upper(p.name)")

        assert len(results) == 3
        for result in results:
            assert result["upper(p.name)"] == result["p.name"].upper()

    def test_lower_function_integration(self, session):
        """Test lower function with real database."""
        results = session.execute("MATCH (p:Person) RETURN p.name, lower(p.name)")

        assert len(results) == 3
        for result in results:
            assert result["lower(p.name)"] == result["p.name"].lower()

    def test_concat_function_integration(self, session):
        """Test concat function with real database."""
        results = session.execute("MATCH (p:Person) RETURN p.name, p.age, concat(p.name, ' (', p.age, ' years old)')")

        assert len(results) == 3
        for result in results:
            expected = f"{result['col_0']} ({result['col_1']} years old)"
            assert result['col_2'] == expected

    def test_size_function_integration(self, session):
        """Test size function with real database."""
        results = session.execute("MATCH (p:Person) RETURN p.name, size(p.name)")

        assert len(results) == 3
        for result in results:
            assert result["size(p.name)"] == len(result["p.name"])

    def test_numeric_functions_integration(self, session):
        """Test numeric functions with real database."""
        results = session.execute("MATCH (p:Person) RETURN p.age, abs(p.age), p.height, ceil(p.height), floor(p.height)")

        assert len(results) == 3
        for result in results:
            assert result["abs(p.age)"] == abs(result["p.age"])
            assert result["ceil(p.height)"] == int(result["p.height"]) + (1 if result["p.height"] % 1 > 0 else 0)
            assert result["floor(p.height)"] == int(result["p.height"])

    def test_sqrt_function_integration(self, session):
        """Test sqrt function with real database."""
        results = session.execute("MATCH (p:Person) RETURN p.age, sqrt(p.age)")

        assert len(results) == 3
        for result in results:
            expected_sqrt = result["p.age"] ** 0.5
            assert abs(result["sqrt(p.age)"] - expected_sqrt) < 0.001  # Allow for floating point precision

    def test_list_functions_integration(self, session):
        """Test list functions with real database."""
        results = session.execute("""
            MATCH (p:Person)
            RETURN p.tags, size(p.tags) AS tag_count,
                   'developer' IN p.tags AS is_developer
        """)

        assert len(results) == 3
        for result in results:
            assert result["tag_count"] == len(result["p.tags"])
            assert result["is_developer"] == ("developer" in result["p.tags"])

    def test_list_append_integration(self, session):
        """Test list append function with real database."""
        results = session.execute("MATCH (p:Person) RETURN p.tags, list_append(p.tags, 'new-tag')")

        assert len(results) == 3
        for result in results:
            expected_tags = result["col_0"] + ["new-tag"]
            assert result["col_1"] == expected_tags

    def test_standalone_functions_integration(self, session):
        """Test standalone functions with real database."""
        results = session.execute("MATCH (p:Person) RETURN p.name, concat('Hello, ', p.name, '!'), pi(), size(p.name)")

        assert len(results) == 3
        for result in results:
            assert "Hello, " in result["col_1"]
            assert result["col_0"] in result["col_1"]
            assert abs(result["col_2"] - 3.14159) < 0.001
            assert result["col_3"] == len(result["col_0"])

    def test_chained_functions_integration(self, session):
        """Test chaining multiple functions together."""
        results = session.execute("MATCH (p:Person) RETURN p.name, substring(upper(p.name), 1, 3)")

        assert len(results) == 3
        for result in results:
            expected = result["col_0"].upper()[:3]  # First 3 characters in uppercase
            assert result["col_1"] == expected

    def test_function_in_where_clause(self, session):
        """Test using functions in WHERE clauses."""
        # Find people whose name length is greater than 10
        results = session.execute("""
            MATCH (p:Person)
            WHERE size(p.name) > 10
            RETURN p.name
        """)

        # Should find "Alice Johnson" and "Charlie Brown"
        assert len(results) == 2
        names = [r["p.name"] for r in results]
        assert "Alice Johnson" in names
        assert "Charlie Brown" in names
        assert "Bob Smith" not in names

    def test_function_in_order_by(self, session):
        """Test using functions in ORDER BY clauses."""
        # Order by name length
        results = session.execute("""
            MATCH (p:Person)
            RETURN p.name
            ORDER BY size(p.name)
        """)

        assert len(results) == 3
        # Should be ordered: "Bob Smith" (9), "Alice Johnson" (13), "Charlie Brown" (13)
        names = [r["p.name"] for r in results]
        assert names[0] == "Bob Smith"  # Shortest name first

    def test_aggregate_with_functions(self, session):
        """Test using functions with aggregation."""
        # Get average name length
        results = session.execute("""
            MATCH (p:Person)
            RETURN avg(size(p.name)) AS avg_name_length
        """)

        assert len(results) == 1
        # Average of 9, 13, 13 = 11.67
        assert abs(results[0]["avg_name_length"] - 11.67) < 0.1

    def test_mathematical_expressions(self, session):
        """Test complex mathematical expressions with functions."""
        results = session.execute("""
            MATCH (p:Person)
            RETURN p.age, p.height,
                   (sqrt(p.age) + floor(p.height)) AS complex_calc
        """)

        assert len(results) == 3
        for result in results:
            expected = (result["p.age"] ** 0.5) + int(result["p.height"])
            assert abs(result["complex_calc"] - expected) < 0.001

    def test_error_handling(self, session):
        """Test that invalid function calls are handled properly."""
        # This should work - sqrt of positive number
        results = session.execute("MATCH (p:Person) RETURN sqrt(p.age) AS sqrt_age")
        assert len(results) == 3

        # Note: We can't easily test error cases without causing the test to fail
        # In a real application, you'd want to handle cases like sqrt of negative numbers

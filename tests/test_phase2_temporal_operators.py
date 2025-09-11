# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for Date/Time/Interval Operators Implementation.
"""

import pytest

from kuzualchemy.kuzu_query_fields import QueryField
from kuzualchemy.kuzu_query_expressions import TemporalExpression, TemporalOperator


class TestTemporalOperatorsImplementation:
    """Tests for all 13 temporal operators."""
    
    # ============================================================================
    # DATE OPERATORS TESTS (4 operators)
    # ============================================================================
    
    def test_date_add_int64(self):
        """Test DATE + INT64 operator."""
        date_field = QueryField("created_date")
        expr = date_field.date_add(5)
        
        # Validate expression type and structure
        assert isinstance(expr, TemporalExpression)
        assert expr.operator == TemporalOperator.DATE_ADD_INT
        assert expr.left == date_field
        assert expr.right == 5
        
        # Validate Cypher generation
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "(n.created_date + 5)"
    
    def test_date_add_interval(self):
        """Test DATE + INTERVAL operator."""
        date_field = QueryField("start_date")
        interval = "INTERVAL('3 DAYS')"
        expr = date_field.date_add(interval)
        
        assert isinstance(expr, TemporalExpression)
        assert expr.operator == TemporalOperator.DATE_ADD_INTERVAL
        assert expr.left == date_field
        assert expr.right == interval
        
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "(n.start_date + 'INTERVAL('3 DAYS')')"
    
    def test_date_sub_date(self):
        """Test DATE - DATE operator."""
        date_field1 = QueryField("end_date")
        date_field2 = QueryField("start_date")
        expr = date_field1.date_sub(date_field2)
        
        assert isinstance(expr, TemporalExpression)
        assert expr.operator == TemporalOperator.DATE_SUB_DATE
        assert expr.left == date_field1
        assert expr.right == date_field2
        
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "(n.end_date - n.start_date)"
    
    def test_date_sub_interval(self):
        """Test DATE - INTERVAL operator."""
        date_field = QueryField("deadline")
        interval = "INTERVAL('10 DAYS')"
        expr = date_field.date_sub(interval)
        
        assert isinstance(expr, TemporalExpression)
        assert expr.operator == TemporalOperator.DATE_SUB_INTERVAL
        assert expr.left == date_field
        assert expr.right == interval
        
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "(n.deadline - 'INTERVAL('10 DAYS')')"
    
    # ============================================================================
    # TIMESTAMP OPERATORS TESTS (3 operators)
    # ============================================================================
    
    def test_timestamp_add_interval(self):
        """Test TIMESTAMP + INTERVAL operator."""
        timestamp_field = QueryField("created_at")
        interval = "INTERVAL('4 minutes, 3 hours, 2 days')"
        expr = timestamp_field.timestamp_add(interval)
        
        assert isinstance(expr, TemporalExpression)
        assert expr.operator == TemporalOperator.TIMESTAMP_ADD_INTERVAL
        assert expr.left == timestamp_field
        assert expr.right == interval
        
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "(n.created_at + 'INTERVAL('4 minutes, 3 hours, 2 days')')"
    
    def test_timestamp_sub_timestamp(self):
        """Test TIMESTAMP - TIMESTAMP operator."""
        timestamp_field1 = QueryField("updated_at")
        timestamp_field2 = QueryField("created_at")
        expr = timestamp_field1.timestamp_sub(timestamp_field2)
        
        assert isinstance(expr, TemporalExpression)
        assert expr.operator == TemporalOperator.TIMESTAMP_SUB_TIMESTAMP
        assert expr.left == timestamp_field1
        assert expr.right == timestamp_field2
        
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "(n.updated_at - n.created_at)"
    
    def test_timestamp_sub_interval(self):
        """Test TIMESTAMP - INTERVAL operator."""
        timestamp_field = QueryField("event_time")
        interval = "INTERVAL('35 days 2 years 3 hours')"
        expr = timestamp_field.timestamp_sub(interval)
        
        assert isinstance(expr, TemporalExpression)
        assert expr.operator == TemporalOperator.TIMESTAMP_SUB_INTERVAL
        assert expr.left == timestamp_field
        assert expr.right == interval
        
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "(n.event_time - 'INTERVAL('35 days 2 years 3 hours')')"
    
    # ============================================================================
    # INTERVAL OPERATORS TESTS (6 operators)
    # ============================================================================
    
    def test_interval_add_interval(self):
        """Test INTERVAL + INTERVAL operator."""
        interval_field = QueryField("duration1")
        interval2 = "INTERVAL('20 MILLISECONDS 30 HOURS 20 DAYS')"
        expr = interval_field.interval_add(interval2)
        
        assert isinstance(expr, TemporalExpression)
        assert expr.operator == TemporalOperator.INTERVAL_ADD_INTERVAL
        assert expr.left == interval_field
        assert expr.right == interval2
        
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "(n.duration1 + 'INTERVAL('20 MILLISECONDS 30 HOURS 20 DAYS')')"
    
    def test_interval_add_date(self):
        """Test INTERVAL + DATE operator."""
        interval_field = QueryField("offset")
        date_str = "DATE('2025-10-01')"
        expr = interval_field.interval_add(date_str)
        
        assert isinstance(expr, TemporalExpression)
        assert expr.operator == TemporalOperator.INTERVAL_ADD_DATE
        assert expr.left == interval_field
        assert expr.right == date_str
        
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "(n.offset + 'DATE('2025-10-01')')"
    
    def test_interval_add_timestamp(self):
        """Test INTERVAL + TIMESTAMP operator."""
        interval_field = QueryField("delay")
        timestamp_str = "TIMESTAMP('2013-02-21')"
        expr = interval_field.interval_add(timestamp_str)
        
        assert isinstance(expr, TemporalExpression)
        assert expr.operator == TemporalOperator.INTERVAL_ADD_TIMESTAMP
        assert expr.left == interval_field
        assert expr.right == timestamp_str
        
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "(n.delay + 'TIMESTAMP('2013-02-21')')"
    
    def test_interval_sub_interval(self):
        """Test INTERVAL - INTERVAL operator."""
        interval_field1 = QueryField("total_duration")
        interval2 = "INTERVAL('1 DAYS')"
        expr = interval_field1.interval_sub(interval2)
        
        assert isinstance(expr, TemporalExpression)
        assert expr.operator == TemporalOperator.INTERVAL_SUB_INTERVAL
        assert expr.left == interval_field1
        assert expr.right == interval2
        
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "(n.total_duration - 'INTERVAL('1 DAYS')')"
    
    # ============================================================================
    # COMPLEX TEMPORAL EXPRESSIONS TESTS
    # ============================================================================
    
    def test_nested_temporal_expressions(self):
        """Test nested temporal expressions."""
        date_field = QueryField("start_date")
        timestamp_field = QueryField("created_at")
        
        # Test complex expression: (date + 5) - INTERVAL('1 DAY')
        expr1 = date_field.date_add(5)
        expr2 = expr1.date_sub("INTERVAL('1 DAY')")
        
        cypher = expr2.to_cypher({"n": "n"})
        assert cypher == "((n.start_date + 5) - 'INTERVAL('1 DAY')')"
    
    def test_temporal_with_arithmetic_operations(self):
        """Test temporal expressions combined with arithmetic operations."""
        date_field = QueryField("event_date")
        
        # Test temporal expression in arithmetic context
        temporal_expr = date_field.date_add(10)
        arithmetic_expr = temporal_expr + 5  # This should create an ArithmeticExpression
        
        cypher = arithmetic_expr.to_cypher({"n": "n"})
        assert cypher == "((n.event_date + 10) + 5)"
    
    def test_temporal_parameter_handling(self):
        """Test parameter handling in temporal expressions."""
        date_field = QueryField("created_date")
        dynamic_interval = "dynamic_interval_param"
        
        expr = date_field.date_add(dynamic_interval)
        params = expr.get_parameters()
        
        # Should have one parameter for the dynamic interval
        assert len(params) == 1
        assert dynamic_interval in params.values()
    
    def test_temporal_field_reference_extraction(self):
        """Test field reference extraction from temporal expressions."""
        date_field1 = QueryField("start_date")
        date_field2 = QueryField("end_date")
        
        expr = date_field1.date_sub(date_field2)
        refs = expr.get_field_references()
        
        assert "start_date" in refs
        assert "end_date" in refs
        assert len(refs) == 2
    
    def test_all_temporal_operators_combined(self):
        """Test covering all temporal operator types."""
        # Date operators
        date_field = QueryField("date_col")
        date_add_int = date_field.date_add(7)
        date_add_interval = date_field.date_add("INTERVAL('1 WEEK')")
        date_sub_date = date_field.date_sub(QueryField("other_date"))
        date_sub_interval = date_field.date_sub("INTERVAL('2 DAYS')")
        
        # Timestamp operators
        ts_field = QueryField("timestamp_col")
        ts_add_interval = ts_field.timestamp_add("INTERVAL('1 HOUR')")
        ts_sub_timestamp = ts_field.timestamp_sub(QueryField("other_timestamp"))
        ts_sub_interval = ts_field.timestamp_sub("INTERVAL('30 MINUTES')")
        
        # Interval operators
        interval_field = QueryField("interval_col")
        interval_add_interval = interval_field.interval_add("INTERVAL('1 DAY')")
        interval_add_date = interval_field.interval_add("DATE('2023-01-01')")
        interval_add_timestamp = interval_field.interval_add("TIMESTAMP('2023-01-01 12:00:00')")
        interval_sub_interval = interval_field.interval_sub("INTERVAL('1 HOUR')")
        
        # Validate all expressions generate correct Cypher
        expressions = [
            (date_add_int, "(n.date_col + 7)"),
            (date_add_interval, "(n.date_col + 'INTERVAL('1 WEEK')')"),
            (date_sub_date, "(n.date_col - n.other_date)"),
            (date_sub_interval, "(n.date_col - 'INTERVAL('2 DAYS')')"),
            (ts_add_interval, "(n.timestamp_col + 'INTERVAL('1 HOUR')')"),
            (ts_sub_timestamp, "(n.timestamp_col - n.other_timestamp)"),
            (ts_sub_interval, "(n.timestamp_col - 'INTERVAL('30 MINUTES')')"),
            (interval_add_interval, "(n.interval_col + 'INTERVAL('1 DAY')')"),
            (interval_add_date, "(n.interval_col + 'DATE('2023-01-01')')"),
            (interval_add_timestamp, "(n.interval_col + 'TIMESTAMP('2023-01-01 12:00:00')')"),
            (interval_sub_interval, "(n.interval_col - 'INTERVAL('1 HOUR')')")
        ]
        
        for expr, expected_cypher in expressions:
            cypher = expr.to_cypher({"n": "n"})
            assert cypher == expected_cypher, f"Expected {expected_cypher}, got {cypher}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

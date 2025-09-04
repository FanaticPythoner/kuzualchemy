"""
Integration tests for Numeric Operators Implementation.
"""

import pytest

from kuzualchemy.kuzu_query_fields import QueryField
from kuzualchemy.kuzu_query_expressions import ArithmeticExpression, ArithmeticOperator


class TestNumericOperatorsImplementation:
    """Tests for all 6 numeric operators."""
    
    def test_addition_operator_basic(self):
        """Test addition operator (+) - Basic functionality."""
        field = QueryField("value")
        expr = field + 5
        
        # Validate expression type and structure
        assert isinstance(expr, ArithmeticExpression)
        assert expr.operator == ArithmeticOperator.ADD
        assert expr.left == field
        assert expr.right == 5
        
        # Validate Cypher generation
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "(n.value + 5)"
    
    def test_addition_operator_right_hand(self):
        """Test right-hand addition operator (5 + field)."""
        field = QueryField("value")
        expr = 10 + field
        
        assert isinstance(expr, ArithmeticExpression)
        assert expr.operator == ArithmeticOperator.ADD
        assert expr.left == 10
        assert expr.right == field
        
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "(10 + n.value)"
    
    def test_subtraction_operator_basic(self):
        """Test subtraction operator (-) - Basic functionality."""
        field = QueryField("salary")
        expr = field - 1000
        
        assert isinstance(expr, ArithmeticExpression)
        assert expr.operator == ArithmeticOperator.SUB
        assert expr.left == field
        assert expr.right == 1000
        
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "(n.salary - 1000)"
    
    def test_subtraction_operator_right_hand(self):
        """Test right-hand subtraction operator (1000 - field)."""
        field = QueryField("discount")
        expr = 1000 - field
        
        assert isinstance(expr, ArithmeticExpression)
        assert expr.operator == ArithmeticOperator.SUB
        assert expr.left == 1000
        assert expr.right == field
        
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "(1000 - n.discount)"
    
    def test_multiplication_operator_basic(self):
        """Test multiplication operator (*) - Basic functionality."""
        field = QueryField("price")
        expr = field * 1.2
        
        assert isinstance(expr, ArithmeticExpression)
        assert expr.operator == ArithmeticOperator.MUL
        assert expr.left == field
        assert expr.right == 1.2
        
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "(n.price * 1.2)"
    
    def test_multiplication_operator_right_hand(self):
        """Test right-hand multiplication operator (2 * field)."""
        field = QueryField("quantity")
        expr = 3 * field
        
        assert isinstance(expr, ArithmeticExpression)
        assert expr.operator == ArithmeticOperator.MUL
        assert expr.left == 3
        assert expr.right == field
        
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "(3 * n.quantity)"
    
    def test_division_operator_basic(self):
        """Test division operator (/) - Basic functionality."""
        field = QueryField("total")
        expr = field / 2
        
        assert isinstance(expr, ArithmeticExpression)
        assert expr.operator == ArithmeticOperator.DIV
        assert expr.left == field
        assert expr.right == 2
        
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "(n.total / 2)"
    
    def test_division_operator_right_hand(self):
        """Test right-hand division operator (100 / field)."""
        field = QueryField("divisor")
        expr = 100 / field
        
        assert isinstance(expr, ArithmeticExpression)
        assert expr.operator == ArithmeticOperator.DIV
        assert expr.left == 100
        assert expr.right == field
        
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "(100 / n.divisor)"
    
    def test_modulo_operator_basic(self):
        """Test modulo operator (%) - Basic functionality."""
        field = QueryField("id")
        expr = field % 10
        
        assert isinstance(expr, ArithmeticExpression)
        assert expr.operator == ArithmeticOperator.MOD
        assert expr.left == field
        assert expr.right == 10
        
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "(n.id % 10)"
    
    def test_modulo_operator_right_hand(self):
        """Test right-hand modulo operator (100 % field)."""
        field = QueryField("modulus")
        expr = 100 % field
        
        assert isinstance(expr, ArithmeticExpression)
        assert expr.operator == ArithmeticOperator.MOD
        assert expr.left == 100
        assert expr.right == field
        
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "(100 % n.modulus)"
    
    def test_power_operator_basic(self):
        """Test power operator (^) - Basic functionality."""
        field = QueryField("base")
        expr = field ** 2
        
        assert isinstance(expr, ArithmeticExpression)
        assert expr.operator == ArithmeticOperator.POW
        assert expr.left == field
        assert expr.right == 2
        
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "(n.base ^ 2)"
    
    def test_power_operator_right_hand(self):
        """Test right-hand power operator (2 ** field)."""
        field = QueryField("exponent")
        expr = 2 ** field
        
        assert isinstance(expr, ArithmeticExpression)
        assert expr.operator == ArithmeticOperator.POW
        assert expr.left == 2
        assert expr.right == field
        
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "(2 ^ n.exponent)"
    
    def test_nested_arithmetic_expressions(self):
        """Test nested arithmetic expressions - Complex combinations."""
        field1 = QueryField("a")
        field2 = QueryField("b")
        field3 = QueryField("c")
        
        # Test (a + b) * c
        expr = (field1 + field2) * field3
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "((n.a + n.b) * n.c)"
        
        # Test a + (b * c)
        expr2 = field1 + (field2 * field3)
        cypher2 = expr2.to_cypher({"n": "n"})
        assert cypher2 == "(n.a + (n.b * n.c))"
        
        # Test complex nested: (a + b) / (c - 5)
        expr3 = (field1 + field2) / (field3 - 5)
        cypher3 = expr3.to_cypher({"n": "n"})
        assert cypher3 == "((n.a + n.b) / (n.c - 5))"
    
    def test_arithmetic_with_different_data_types(self):
        """Test arithmetic operations with different numeric data types."""
        field = QueryField("value")
        
        # Integer operations
        int_expr = field + 42
        cypher = int_expr.to_cypher({"n": "n"})
        assert cypher == "(n.value + 42)"
        
        # Float operations
        float_expr = field * 3.14159
        cypher = float_expr.to_cypher({"n": "n"})
        assert cypher == "(n.value * 3.14159)"
        
        # Negative numbers
        neg_expr = field - (-10)
        cypher = neg_expr.to_cypher({"n": "n"})
        assert cypher == "(n.value - -10)"
    
    def test_arithmetic_parameter_handling(self):
        """Test parameter handling in arithmetic expressions."""
        field = QueryField("value")
        param_value = "dynamic_param"
        
        expr = field + param_value
        params = expr.get_parameters()
        
        # Should have one parameter for the non-literal value
        assert len(params) == 1
        assert param_value in params.values()
        
        # Parameter name should be deterministic based on counter
        param_name = [k for k in params.keys() if k.startswith('arith_right_')][0]
        assert param_name in params
        assert params[param_name] == param_value
    
    def test_field_reference_extraction(self):
        """Test field reference extraction from arithmetic expressions."""
        field1 = QueryField("first_field")
        field2 = QueryField("second_field")
        
        expr = field1 + field2
        refs = expr.get_field_references()
        
        assert "first_field" in refs
        assert "second_field" in refs
        assert len(refs) == 2
    
    def test_all_operators_combined(self):
        """Test of all 6 operators in one expression."""
        field = QueryField("x")
        
        # Test all operators: ((x + 1) - 2) * 3 / 4 % 5 ^ 2
        expr = ((((field + 1) - 2) * 3) / 4) % 5
        final_expr = expr ** 2
        
        cypher = final_expr.to_cypher({"n": "n"})
        expected = "((((((n.x + 1) - 2) * 3) / 4) % 5) ^ 2)"
        assert cypher == expected
    
    def test_operator_precedence_validation(self):
        """Test that operator precedence is handled correctly through parentheses."""
        field = QueryField("value")

        # Python evaluates 2 * 3 to 6 before creating the expression
        expr = field + 2 * 3  # Python computes 2*3=6, so this becomes field + 6
        cypher = expr.to_cypher({"n": "n"})
        assert cypher == "(n.value + 6)"

        # With explicit parentheses
        expr2 = (field + 2) * 3  # Should be ((field + 2) * 3)
        cypher2 = expr2.to_cypher({"n": "n"})
        assert cypher2 == "((n.value + 2) * 3)"

        # To test actual nested expressions, use fields
        field2 = QueryField("multiplier")
        expr3 = field + (field2 * 3)  # Should be (field + (field2 * 3))
        cypher3 = expr3.to_cypher({"n": "n"})
        assert cypher3 == "(n.value + (n.multiplier * 3))"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

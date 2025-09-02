"""
Tests for numeric functions in KuzuAlchemy.
"""

import pytest
from kuzualchemy.kuzu_query_fields import QueryField
from kuzualchemy.kuzu_functions import pi, ceil, floor, sqrt, sin, cos, tan, ln, log, log2
from kuzualchemy.kuzu_functions import abs as kuzu_abs, pow as kuzu_pow
from kuzualchemy.kuzu_query_expressions import FunctionExpression


class TestNumericFunctions:
    """Test numeric functions on QueryField objects."""

    def test_abs_method(self):
        """Test abs method on QueryField."""
        field = QueryField("value")
        result = field.abs()
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "abs"
        assert len(result.args) == 1
        assert result.args[0] == field

    def test_trigonometric_methods(self):
        """Test trigonometric methods."""
        field = QueryField("angle")
        
        # Test sin
        result = field.sin()
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "sin"
        assert len(result.args) == 1
        assert result.args[0] == field
        
        # Test cos
        result = field.cos()
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "cos"
        assert len(result.args) == 1
        assert result.args[0] == field
        
        # Test tan
        result = field.tan()
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "tan"
        assert len(result.args) == 1
        assert result.args[0] == field

    def test_inverse_trigonometric_methods(self):
        """Test inverse trigonometric methods."""
        field = QueryField("value")
        
        # Test asin
        result = field.asin()
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "asin"
        assert len(result.args) == 1
        assert result.args[0] == field
        
        # Test acos
        result = field.acos()
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "acos"
        assert len(result.args) == 1
        assert result.args[0] == field
        
        # Test atan
        result = field.atan()
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "atan"
        assert len(result.args) == 1
        assert result.args[0] == field

    def test_atan2_method(self):
        """Test atan2 method."""
        field = QueryField("x")
        result = field.atan2(5.0)
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "atan2"
        assert len(result.args) == 2
        assert result.args[0] == field
        assert result.args[1] == 5.0

    def test_rounding_methods(self):
        """Test rounding methods."""
        field = QueryField("value")
        
        # Test ceil
        result = field.ceil()
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "ceil"
        assert len(result.args) == 1
        assert result.args[0] == field
        
        # Test ceiling (alias)
        result = field.ceiling()
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "ceiling"
        assert len(result.args) == 1
        assert result.args[0] == field
        
        # Test floor
        result = field.floor()
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "floor"
        assert len(result.args) == 1
        assert result.args[0] == field

    def test_round_method(self):
        """Test round method."""
        field = QueryField("value")
        
        # Test with default precision
        result = field.round()
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "round"
        assert len(result.args) == 2
        assert result.args[0] == field
        assert result.args[1] == 0
        
        # Test with custom precision
        result = field.round(2)
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "round"
        assert len(result.args) == 2
        assert result.args[0] == field
        assert result.args[1] == 2

    def test_logarithmic_methods(self):
        """Test logarithmic methods."""
        field = QueryField("value")
        
        # Test ln
        result = field.ln()
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "ln"
        assert len(result.args) == 1
        assert result.args[0] == field
        
        # Test log
        result = field.log()
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "log"
        assert len(result.args) == 1
        assert result.args[0] == field
        
        # Test log2
        result = field.log2()
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "log2"
        assert len(result.args) == 1
        assert result.args[0] == field
        
        # Test log10 (alias)
        result = field.log10()
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "log10"
        assert len(result.args) == 1
        assert result.args[0] == field

    def test_power_and_root_methods(self):
        """Test power and root methods."""
        field = QueryField("value")
        
        # Test pow
        result = field.pow(3)
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "pow"
        assert len(result.args) == 2
        assert result.args[0] == field
        assert result.args[1] == 3
        
        # Test sqrt
        result = field.sqrt()
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "sqrt"
        assert len(result.args) == 1
        assert result.args[0] == field

    def test_angle_conversion_methods(self):
        """Test angle conversion methods."""
        field = QueryField("angle")
        
        # Test degrees
        result = field.degrees()
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "degrees"
        assert len(result.args) == 1
        assert result.args[0] == field
        
        # Test radians
        result = field.radians()
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "radians"
        assert len(result.args) == 1
        assert result.args[0] == field

    def test_special_numeric_methods(self):
        """Test special numeric methods."""
        field = QueryField("value")
        
        # Test sign
        result = field.sign()
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "sign"
        assert len(result.args) == 1
        assert result.args[0] == field
        
        # Test negate
        result = field.negate()
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "negate"
        assert len(result.args) == 1
        assert result.args[0] == field
        
        # Test factorial
        result = field.factorial()
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "factorial"
        assert len(result.args) == 1
        assert result.args[0] == field
        
        # Test even
        result = field.even()
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "even"
        assert len(result.args) == 1
        assert result.args[0] == field

    def test_gamma_methods(self):
        """Test gamma function methods."""
        field = QueryField("value")
        
        # Test gamma
        result = field.gamma()
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "gamma"
        assert len(result.args) == 1
        assert result.args[0] == field
        
        # Test lgamma
        result = field.lgamma()
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "lgamma"
        assert len(result.args) == 1
        assert result.args[0] == field

    def test_cotangent_method(self):
        """Test cotangent method."""
        field = QueryField("angle")
        result = field.cot()
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "cot"
        assert len(result.args) == 1
        assert result.args[0] == field

    def test_bitwise_xor_method(self):
        """Test bitwise XOR method."""
        field = QueryField("value")
        result = field.bitwise_xor(7)
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "bitwise_xor"
        assert len(result.args) == 2
        assert result.args[0] == field
        assert result.args[1] == 7


class TestStandaloneNumericFunctions:
    """Test standalone numeric functions."""

    def test_pi_function(self):
        """Test pi function."""
        result = pi()
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "pi"
        assert len(result.args) == 0

    def test_abs_function(self):
        """Test standalone abs function."""
        result = kuzu_abs(-5.5)

        assert isinstance(result, FunctionExpression)
        assert result.function_name == "abs"
        assert len(result.args) == 1
        assert result.args[0] == -5.5

    def test_ceil_function(self):
        """Test standalone ceil function."""
        result = ceil(4.2)
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "ceil"
        assert len(result.args) == 1
        assert result.args[0] == 4.2

    def test_floor_function(self):
        """Test standalone floor function."""
        result = floor(4.8)
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "floor"
        assert len(result.args) == 1
        assert result.args[0] == 4.8

    def test_sqrt_function(self):
        """Test standalone sqrt function."""
        result = sqrt(16)
        
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "sqrt"
        assert len(result.args) == 1
        assert result.args[0] == 16

    def test_pow_function(self):
        """Test standalone pow function."""
        result = kuzu_pow(2, 8)

        assert isinstance(result, FunctionExpression)
        assert result.function_name == "pow"
        assert len(result.args) == 2
        assert result.args[0] == 2
        assert result.args[1] == 8

    def test_trigonometric_functions(self):
        """Test standalone trigonometric functions."""
        # Test sin
        result = sin(1.57)
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "sin"
        assert len(result.args) == 1
        assert result.args[0] == 1.57
        
        # Test cos
        result = cos(0)
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "cos"
        assert len(result.args) == 1
        assert result.args[0] == 0
        
        # Test tan
        result = tan(0.785)
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "tan"
        assert len(result.args) == 1
        assert result.args[0] == 0.785

    def test_logarithmic_functions(self):
        """Test standalone logarithmic functions."""
        # Test ln
        result = ln(2.718)
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "ln"
        assert len(result.args) == 1
        assert result.args[0] == 2.718
        
        # Test log
        result = log(100)
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "log"
        assert len(result.args) == 1
        assert result.args[0] == 100
        
        # Test log2
        result = log2(8)
        assert isinstance(result, FunctionExpression)
        assert result.function_name == "log2"
        assert len(result.args) == 1
        assert result.args[0] == 8


class TestCypherGeneration:
    """Test Cypher generation for numeric functions."""

    def test_simple_function_cypher(self):
        """Test simple function generates correct Cypher."""
        field = QueryField("value")
        result = field.abs()
        
        alias_map = {"n": "n"}
        cypher = result.to_cypher(alias_map)
        
        assert cypher == "abs(n.value)"

    def test_function_with_parameter_cypher(self):
        """Test function with parameter generates correct Cypher."""
        field = QueryField("value")
        result = field.pow(3)
        
        alias_map = {"n": "n"}
        cypher = result.to_cypher(alias_map)
        
        assert cypher == "pow(n.value, 3)"

    def test_standalone_function_cypher(self):
        """Test standalone function generates correct Cypher."""
        result = pi()
        
        alias_map = {"n": "n"}
        cypher = result.to_cypher(alias_map)
        
        assert cypher == "pi()"

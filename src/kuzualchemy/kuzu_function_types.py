"""
Function type classes for Kuzu default value functions.

This module provides the type hierarchy for categorizing Kuzu functions
using pure inheritance. Each function type has its own marker class.
"""

from __future__ import annotations

from typing import Any


# ============================================================================
# BASE FUNCTION TYPES
# ============================================================================

class DefaultFunctionBase:
    """Base class for all default function types."""

    def __init__(self, value: str):
        """
        Initialize a default function.

        :param value: The function string value
        :type value: str
        """
        self.value = value

    def __str__(self) -> str:
        """Return the function string."""
        return self.value

    def __deepcopy__(self, memo):
        """Support for deepcopy operations."""
        return self.__class__(self.value)

    def __copy__(self):
        """Support for copy operations."""
        return self.__class__(self.value)

    def __eq__(self, other):
        """Equality comparison."""
        if not isinstance(other, self.__class__):
            return False
        return self.value == other.value

    def __hash__(self):
        """Hash support for use in sets and as dict keys."""
        return hash((self.__class__.__name__, self.value))


# ============================================================================
# SPECIFIC FUNCTION TYPE MARKERS
# ============================================================================

class TimeFunction(DefaultFunctionBase):
    """
    Time-related function marker class.
    
    Functions of this type handle temporal operations like
    current_timestamp(), current_date(), now(), etc.
    """
    pass


class UUIDFunction(DefaultFunctionBase):
    """
    UUID generation function marker class.
    
    Functions of this type generate unique identifiers.
    """
    pass


class SequenceFunction(DefaultFunctionBase):
    """
    Sequence-based function marker class.
    
    Functions of this type work with database sequences
    for auto-incrementing values.
    """
    
    def with_args(self, *args: Any) -> str:
        """
        Return function with arguments.
        
        :param args: Arguments for the function
        :type args: Any
        :return: Function string with arguments
        :rtype: str
        :raises ValueError: If no arguments provided
        """
        if not args:
            raise ValueError("NEXTVAL requires a sequence name argument")
        return f"{self.value}('{args[0]}')"

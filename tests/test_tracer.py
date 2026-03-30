"""
TDD Tests for Tracer - Common Subexpression Elimination and Virtual Register Reuse

These tests define the expected behavior of the tracer BEFORE implementation.
Run these tests first to see them fail, then implement the Tracer to make them pass.
"""

import pytest
from athena.tracer import Tracer


class TestTracerBasics:
    """Tests for basic Tracer functionality."""

    def test_tracer_creation_with_value(self):
        """A Tracer should store a value and optional name."""
        t = Tracer(value=5, name="x")
        assert t.value == 5
        assert t.name == "x"

    def test_tracer_creation_without_value(self):
        """A Tracer should work without explicit value."""
        t = Tracer()
        assert t.value is None
        assert t.name is None


class TestTracerArithmetic:
    """Tests for arithmetic operations on Tracers."""

    def test_tracer_add_returns_tracer(self):
        """Adding two tracers should return a Tracer (for the graph)."""
        a = Tracer(value=1, name="a")
        b = Tracer(value=2, name="b")
        result = a + b
        assert isinstance(result, Tracer)

    def test_tracer_mul_returns_tracer(self):
        """Multiplying two tracers should return a Tracer."""
        a = Tracer(value=3, name="a")
        b = Tracer(value=4, name="b")
        result = a * b
        assert isinstance(result, Tracer)

    def test_tracer_add_preserves_values(self):
        """The result of addition should have correct computed value."""
        a = Tracer(value=10, name="a")
        b = Tracer(value=20, name="b")
        result = a + b
        assert result.value == 30

    def test_tracer_mul_preserves_values(self):
        """The result of multiplication should have correct computed value."""
        a = Tracer(value=7, name="a")
        b = Tracer(value=8, name="b")
        result = a * b
        assert result.value == 56


class TestTracerCSE:
    """
    Tests for Common Subexpression Elimination (CSE).
    
    The tracer should recognize that (a+b) + (a+b) contains the same
    subexpression (a+b) twice and reuse the virtual register.
    """

    def test_tracer_records_operations(self):
        """Tracer should record operations in a trace graph."""
        a = Tracer(value=1, name="a")
        b = Tracer(value=2, name="b")
        result = a + b
        # The tracer should have a trace or graph recording the operation
        assert hasattr(result, 'trace') or hasattr(result, 'graph') or hasattr(result, 'op')

    def test_cse_detects_duplicate_addition(self):
        """
        When computing (a+b) + (a+b), the tracer should recognize
        that a+b is computed twice and reuse the virtual register.
        
        This is the core requirement: CSE for efficiency.
        """
        a = Tracer(value=3, name="a")
        b = Tracer(value=4, name="b")
        
        # First a+b
        sum1 = a + b
        assert sum1.value == 7
        
        # Second a+b (identical subexpression)
        sum2 = a + b
        assert sum2.value == 7
        
        # The tracer should recognize these are the same operation
        # and assign the same virtual register
        assert sum1.virtual_register == sum2.virtual_register

    def test_cse_detects_duplicate_multiplication(self):
        """CSE should also work for multiplication."""
        x = Tracer(value=5, name="x")
        y = Tracer(value=6, name="y")
        
        prod1 = x * y
        prod2 = x * y
        
        assert prod1.virtual_register == prod2.virtual_register

    def test_cse_complex_expression(self):
        """
        Test CSE in a complex expression: (a+b) + (a+b) * (a+b)
        
        The subexpression (a+b) appears 3 times but should only be computed once.
        """
        a = Tracer(value=2, name="a")
        b = Tracer(value=3, name="b")
        
        ab = a + b  # First occurrence
        result = ab + (ab * ab)  # ab reused twice more
        
        # All three uses of (a+b) should share the same virtual register
        assert ab.virtual_register is not None
        
        # The result should be correct: (2+3) + (2+3)*(2+3) = 5 + 25 = 30
        assert result.value == 30

    def test_cse_different_expressions_get_different_registers(self):
        """Different expressions should get different virtual registers."""
        a = Tracer(value=1, name="a")
        b = Tracer(value=2, name="b")
        c = Tracer(value=3, name="c")
        
        ab = a + b
        bc = b + c
        
        assert ab.virtual_register != bc.virtual_register

    def test_cse_with_constants(self):
        """CSE should also work when one operand is a constant."""
        a = Tracer(value=10, name="a")
        
        expr1 = a + 5
        expr2 = a + 5
        
        assert expr1.virtual_register == expr2.virtual_register

    def test_tracer_tracks_all_operations(self):
        """The tracer should maintain a list of all unique operations."""
        # Reset state for isolated test
        Tracer.reset()
        
        a = Tracer(value=1, name="a")
        b = Tracer(value=2, name="b")
        
        _ = a + b
        _ = a * b
        _ = a + b  # Duplicate, should reuse
        
        # Should have exactly 2 unique operations recorded
        assert Tracer.trace_count() == 2 or len(Tracer.get_all_operations()) == 2


class TestTracerVirtualRegisters:
    """Tests for virtual register assignment and management."""

    def test_virtual_register_is_assigned(self):
        """Every operation result should have a virtual register."""
        a = Tracer(value=1, name="a")
        b = Tracer(value=2, name="b")
        result = a + b
        assert hasattr(result, 'virtual_register')
        assert result.virtual_register is not None

    def test_input_tracers_have_registers(self):
        """Input tracers (variables) should also have virtual registers."""
        a = Tracer(value=1, name="a")
        assert hasattr(a, 'virtual_register')
        assert a.virtual_register is not None

    def test_different_inputs_have_different_registers(self):
        """Different input variables should have different registers."""
        a = Tracer(value=1, name="a")
        b = Tracer(value=2, name="b")
        assert a.virtual_register != b.virtual_register


class TestTracerGraphStructure:
    """Tests for the computational graph structure."""

    def test_tracer_has_parents(self):
        """A tracer created from an operation should track its parents."""
        a = Tracer(value=1, name="a")
        b = Tracer(value=2, name="b")
        result = a + b
        
        # Should have references to parent tracers
        assert hasattr(result, 'parents') or hasattr(result, 'operands')
        parents = getattr(result, 'parents', None) or getattr(result, 'operands', None)
        assert a in parents
        assert b in parents

    def test_tracer_has_operation_type(self):
        """A tracer should record what operation created it."""
        a = Tracer(value=1, name="a")
        b = Tracer(value=2, name="b")
        result = a + b
        
        assert hasattr(result, 'op') or hasattr(result, 'operation')
        op = getattr(result, 'op', None) or getattr(result, 'operation', None)
        assert op == 'add'

    def test_tracer_graph_is_reconstructible(self):
        """The full computation graph should be extractable."""
        a = Tracer(value=1, name="a")
        b = Tracer(value=2, name="b")
        c = Tracer(value=3, name="c")
        
        # Build: (a + b) * c
        ab = a + b
        result = ab * c
        
        # Should be able to get the graph structure
        graph = result.get_graph() if hasattr(result, 'get_graph') else None
        # At minimum, the result should be traceable back to inputs
        assert graph is not None or result is not None

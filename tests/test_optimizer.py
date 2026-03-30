"""
TDD Tests for Optimizer - Constant Folding and Operator Fusion

These tests define the expected behavior of the optimizer BEFORE implementation.
Run these tests first to see them fail, then implement the Optimizer to make them pass.
"""

import pytest
from athena.tracer import Tracer
from athena.optimizer import Optimizer


class TestOptimizerConstantFolding:
    """Tests for constant folding optimization."""

    def test_fold_add_constants(self):
        """
        2 + 3 should be folded to 5 at compile time.
        """
        Tracer.reset()
        
        # Create a graph: 2 + 3
        a = Tracer(value=2)
        b = Tracer(value=3)
        result = a + b
        
        optimizer = Optimizer()
        optimized = optimizer.optimize(result)
        
        # The result should be a constant 5, not an add operation
        assert optimized.value == 5
        assert optimized.op is None or optimized.op == 'const'

    def test_fold_mul_constants(self):
        """
        4 * 5 should be folded to 20 at compile time.
        """
        Tracer.reset()
        
        a = Tracer(value=4)
        b = Tracer(value=5)
        result = a * b
        
        optimizer = Optimizer()
        optimized = optimizer.optimize(result)
        
        assert optimized.value == 20

    def test_fold_nested_constants(self):
        """
        (1 + 2) + (3 + 4) should fold to 10.
        """
        Tracer.reset()
        
        # Build: (1+2) + (3+4)
        a = Tracer(value=1)
        b = Tracer(value=2)
        c = Tracer(value=3)
        d = Tracer(value=4)
        
        ab = a + b  # 3
        cd = c + d  # 7
        result = ab + cd  # 10
        
        optimizer = Optimizer()
        optimized = optimizer.optimize(result)
        
        assert optimized.value == 10
        # Should be fully folded (no operations left)
        assert optimized.op is None or optimized.op == 'const'

    def test_fold_mixed_constants_and_vars(self):
        """
        x + 0 should simplify to x.
        x * 1 should simplify to x.
        x * 0 should simplify to 0.
        """
        Tracer.reset()
        
        x = Tracer(value=7, name="x")
        
        # x + 0 = x
        result = x + 0
        optimizer = Optimizer()
        optimized = optimizer.optimize(result)
        assert optimized.virtual_register == x.virtual_register
        
        # x * 1 = x
        result = x * 1
        optimized = optimizer.optimize(result)
        assert optimized.virtual_register == x.virtual_register
        
        # x * 0 = 0
        result = x * 0
        optimized = optimizer.optimize(result)
        assert optimized.value == 0

    def test_no_fold_with_variables(self):
        """
        x + y should NOT be folded (variables are not constants).
        """
        Tracer.reset()
        
        x = Tracer(value=5, name="x")
        y = Tracer(value=3, name="y")
        result = x + y
        
        optimizer = Optimizer()
        optimized = optimizer.optimize(result)
        
        # Should still be an add operation
        assert optimized.op == 'add'
        assert optimized.value == 8  # value is preserved for runtime


class TestOptimizerCSE:
    """Tests for CSE at the optimizer level (post-tracing optimization)."""

    def test_optimizer_removes_duplicate_ops(self):
        """
        If tracer already did CSE, optimizer should preserve it.
        This tests that optimization doesn't break CSE.
        """
        Tracer.reset()
        
        a = Tracer(value=2, name="a")
        b = Tracer(value=3, name="b")
        
        # Build: (a+b) + (a+b)
        ab1 = a + b
        ab2 = a + b  # Should share VR with ab1
        result = ab1 + ab2
        
        optimizer = Optimizer()
        optimized = optimizer.optimize(result)
        
        # The graph should still work correctly
        assert optimized.value == 10  # (2+3) + (2+3) = 10

    def test_optimizer_produces_valid_graph(self):
        """
        After optimization, the graph should still be valid and executable.
        """
        Tracer.reset()
        
        a = Tracer(value=10, name="a")
        b = Tracer(value=20, name="b")
        c = Tracer(value=30, name="c")
        
        # a + b * c
        bc = b * c
        result = a + bc
        
        optimizer = Optimizer()
        optimized = optimizer.optimize(result)
        
        # (10 + (20 * 30)) = 10 + 600 = 610
        assert optimized.value == 610


class TestOptimizerFusion:
    """Tests for operator fusion (combining operations)."""

    def test_fuse_chain_of_adds(self):
        """
        a + b + c should be fusible into a single operation.
        """
        Tracer.reset()
        
        a = Tracer(value=1, name="a")
        b = Tracer(value=2, name="b")
        c = Tracer(value=3, name="c")
        
        # a + b + c
        ab = a + b
        result = ab + c
        
        optimizer = Optimizer()
        optimized = optimizer.optimize(result)
        
        # Value should be correct
        assert optimized.value == 6
        
        # After fusion, might be a single fused op or still separate
        # The key is it works correctly
        assert optimized is not None

    def test_fuse_chain_of_muls(self):
        """
        a * b * c should work correctly.
        """
        Tracer.reset()
        
        a = Tracer(value=2, name="a")
        b = Tracer(value=3, name="b")
        c = Tracer(value=4, name="c")
        
        ab = a * b
        result = ab * c
        
        optimizer = Optimizer()
        optimized = optimizer.optimize(result)
        
        # 2 * 3 * 4 = 24
        assert optimized.value == 24


class TestOptimizerIntegration:
    """Integration tests for optimizer with tracer."""

    def test_optimize_complex_expression(self):
        """
        Optimize: (a + b) * (a + b) + c
        Should recognize (a+b) is computed twice.
        """
        Tracer.reset()
        
        a = Tracer(value=2, name="a")
        b = Tracer(value=3, name="b")
        c = Tracer(value=10, name="c")
        
        ab = a + b  # 5
        ab_sq = ab * ab  # 25
        result = ab_sq + c  # 35
        
        optimizer = Optimizer()
        optimized = optimizer.optimize(result)
        
        assert optimized.value == 35

    def test_optimizer_preserves_input_tracers(self):
        """
        Optimization should not remove input tracers (variables).
        """
        Tracer.reset()
        
        x = Tracer(value=5, name="x")
        y = Tracer(value=10, name="y")
        result = x * y + x  # x*y + x = 5*10 + 5 = 55
        
        optimizer = Optimizer()
        optimized = optimizer.optimize(result)
        
        assert optimized.value == 55
        # The graph should reference x and y
        graph = optimized.get_graph()
        assert graph is not None

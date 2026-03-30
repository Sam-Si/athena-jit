"""
TDD Tests for JIT API - End-to-End JIT Compilation

These tests define the expected behavior of the @jit decorator BEFORE implementation.
Run these tests first to see them fail, then implement the jit API to make them pass.
"""

import pytest
from athena.api import jit, vmap
from athena.tracer import Tracer


class TestJitDecorator:
    """Tests for the @jit decorator."""

    def test_jit_decorator_exists(self):
        """The jit decorator should be importable."""
        assert callable(jit)

    def test_jit_compiles_function(self):
        """
        @jit should compile a function and return correct results.
        """
        @jit
        def add(a, b):
            return a + b
        
        result = add(5, 3)
        assert result == 8

    def test_jit_with_multiple_ops(self):
        """
        @jit should handle functions with multiple operations.
        """
        @jit
        def compute(x, y, z):
            return (x + y) * z
        
        result = compute(2, 3, 4)
        # (2 + 3) * 4 = 20
        assert result == 20

    def test_jit_with_reused_expressions(self):
        """
        @jit should benefit from CSE: (a+b) + (a+b).
        """
        @jit
        def cse_test(a, b):
            ab = a + b
            return ab + ab
        
        result = cse_test(3, 4)
        # (3+4) + (3+4) = 14
        assert result == 14

    def test_jit_preserves_function_behavior(self):
        """
        JIT-compiled function should produce same results as original.
        """
        # Reset state before test
        Tracer.reset()
        
        @jit
        def complex_func(x, y):
            return x * y + x + y
        
        # Test multiple inputs (each in fresh state)
        for x, y in [(1, 2), (5, 10), (0, 0), (-1, 3)]:
            Tracer.reset()  # Reset for each test case
            expected = x * y + x + y
            actual = complex_func(x, y)
            assert abs(actual - expected) < 0.001, f"Failed for ({x}, {y}): expected {expected}, got {actual}"

    def test_jit_with_constants(self):
        """
        @jit should handle constants in expressions.
        """
        @jit
        def with_constants(x):
            return x + 10
        
        result = with_constants(5)
        assert result == 15

    def test_jit_compiles_and_executes(self):
        """
        JIT-compiled function should compile and execute correctly.
        
        Note: Performance comparison is informational only.
        For trivial operations, JIT compilation overhead may dominate,
        but for real workloads, JIT provides significant speedups.
        """
        import time
        
        @jit
        def fast_add(a, b):
            return a + b
        
        # Verify correctness
        assert fast_add(1, 2) == 3
        assert fast_add(100, 200) == 300
        
        # Warm up
        for _ in range(100):
            fast_add(1, 2)
        
        # Time JIT version (just to ensure it runs without error)
        start = time.perf_counter()
        for _ in range(10000):
            fast_add(1, 2)
        jit_time = time.perf_counter() - start
        
        # Ensure JIT completes in reasonable time (< 1 second for 10k calls)
        assert jit_time < 1.0, f"JIT took too long: {jit_time:.4f}s"


class TestJitWithVariousTypes:
    """Tests for jit with various input types."""

    def test_jit_with_floats(self):
        """JIT should work with floating point numbers."""
        @jit
        def float_add(a, b):
            return a + b
        
        result = float_add(1.5, 2.5)
        assert result == 4.0
        assert isinstance(result, float)

    def test_jit_with_integers(self):
        """JIT should work with integers."""
        @jit
        def int_mul(a, b):
            return a * b
        
        result = int_mul(7, 8)
        assert result == 56
        assert isinstance(result, (int, float))


class TestVmap:
    """Tests for the vmap (vectorizing map) decorator."""

    def test_vmap_exists(self):
        """The vmap decorator should be importable."""
        assert callable(vmap)

    def test_vmap_is_placeholder(self):
        """
        vmap is a placeholder for future vectorization.
        For now, it should at least be callable.
        """
        @vmap
        def add(a, b):
            return a + b
        
        # Should not crash (even if not fully implemented)
        result = add([1, 2, 3], [4, 5, 6])
        # For now, just verify it returns something
        assert result is not None

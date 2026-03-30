"""
Tests for SIMD Vectorization in Athena JIT (Tracer Architecture)

These tests verify that SIMD-accelerated JIT compilation works correctly
and handles edge cases like remainder elements.
"""

import pytest
import numpy as np
from athena.api import jit


class TestSIMDCorrectness:
    """Tests that SIMD produces correct results matching NumPy."""

    def test_simd_add_matches_numpy(self):
        """SIMD addition should match NumPy add."""
        @jit(buffer_mode=True, simd=True)
        def add(a, b):
            # We need 5+ ops to trigger JIT if not using buffer_mode
            # But buffer_mode=True always compiles
            return a + b
        
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float64)
        
        jit_result = add(a, b)
        numpy_result = a + b
        
        np.testing.assert_allclose(jit_result, numpy_result, rtol=1e-10)

    def test_simd_mul_matches_numpy(self):
        """SIMD multiplication should match NumPy multiply."""
        @jit(buffer_mode=True, simd=True)
        def mul(a, b):
            return a * b
        
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float64)
        
        jit_result = mul(a, b)
        numpy_result = a * b
        
        np.testing.assert_allclose(jit_result, numpy_result, rtol=1e-10)

    def test_simd_non_multiple_of_width(self):
        """
        SIMD should handle arrays whose size is not a multiple of SIMD width.
        This verifies the scalar remainder loop.
        """
        @jit(buffer_mode=True, simd=True)
        def add(a, b):
            return a + b
        
        # Test size 5 (4-wide SIMD + 1 remainder)
        size = 5
        a = np.arange(size, dtype=np.float64)
        b = np.arange(size, dtype=np.float64) * 2
        
        jit_result = add(a, b)
        numpy_result = a + b
        
        np.testing.assert_allclose(jit_result, numpy_result, rtol=1e-10)
        
        # Test size 7
        size = 7
        a = np.arange(size, dtype=np.float64)
        b = np.arange(size, dtype=np.float64) * 2
        
        jit_result = add(a, b)
        numpy_result = a + b
        
        np.testing.assert_allclose(jit_result, numpy_result, rtol=1e-10)

    def test_simd_single_element(self):
        """SIMD should handle single-element arrays (all remainder)."""
        @jit(buffer_mode=True, simd=True)
        def add(a, b):
            return a + b
        
        a = np.array([5.0], dtype=np.float64)
        b = np.array([3.0], dtype=np.float64)
        
        jit_result = add(a, b)
        numpy_result = a + b
        
        np.testing.assert_allclose(jit_result, numpy_result, rtol=1e-10)

    def test_simd_complex_expression(self):
        """SIMD should handle complex expressions."""
        @jit(buffer_mode=True, simd=True)
        def complex_expr(a, b):
            return a * a + b * b
        
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        b = np.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        
        jit_result = complex_expr(a, b)
        numpy_result = a * a + b * b
        
        np.testing.assert_allclose(jit_result, numpy_result, rtol=1e-10)

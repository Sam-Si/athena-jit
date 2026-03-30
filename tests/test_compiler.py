"""
TDD Tests for Compiler - LLVM IR Generation and Execution

These tests define the expected behavior of the compiler BEFORE implementation.
Run these tests first to see them fail, then implement the Compiler to make them pass.
"""

import pytest
from athena.tracer import Tracer
from athena.optimizer import Optimizer
from athena.compiler import AthenaCompiler


class TestCompilerBasics:
    """Tests for basic compiler functionality."""

    def test_compiler_initialization(self):
        """Compiler should initialize LLVM and target machine."""
        compiler = AthenaCompiler()
        assert compiler is not None
        assert compiler.target_machine is not None

    def test_compiler_has_compile_method(self):
        """Compiler should have a compile method."""
        compiler = AthenaCompiler()
        assert hasattr(compiler, 'compile')
        assert callable(compiler.compile)


class TestCompilerIRGeneration:
    """Tests for LLVM IR generation from tracer graphs."""

    def test_compile_simple_add(self):
        """
        Compile (a + b) where a=5, b=3.
        Should generate valid LLVM IR and execute correctly.
        """
        Tracer.reset()
        
        a = Tracer(value=5, name="a")
        b = Tracer(value=3, name="b")
        result = a + b
        
        compiler = AthenaCompiler()
        compiled_func = compiler.compile(result)
        
        # Execute with inputs
        output = compiled_func(5, 3)
        assert output == 8

    def test_compile_simple_mul(self):
        """
        Compile (a * b) where a=4, b=7.
        """
        Tracer.reset()
        
        a = Tracer(value=4, name="a")
        b = Tracer(value=7, name="b")
        result = a * b
        
        compiler = AthenaCompiler()
        compiled_func = compiler.compile(result)
        
        output = compiled_func(4, 7)
        assert output == 28

    def test_compile_chained_ops(self):
        """
        Compile a + b * c.
        Should respect operator precedence (mul before add).
        """
        Tracer.reset()
        
        a = Tracer(value=1, name="a")
        b = Tracer(value=2, name="b")
        c = Tracer(value=3, name="c")
        
        # a + (b * c) = 1 + (2 * 3) = 7
        bc = b * c
        result = a + bc
        
        compiler = AthenaCompiler()
        compiled_func = compiler.compile(result)
        
        output = compiled_func(1, 2, 3)
        assert output == 7

    def test_compile_complex_expression(self):
        """
        Compile: (a + b) * (a + b) + c
        Tests CSE integration - (a+b) should be computed once.
        """
        Tracer.reset()
        
        a = Tracer(value=2, name="a")
        b = Tracer(value=3, name="b")
        c = Tracer(value=10, name="c")
        
        ab = a + b  # 5
        ab_sq = ab * ab  # 25
        result = ab_sq + c  # 35
        
        compiler = AthenaCompiler()
        compiled_func = compiler.compile(result)
        
        output = compiled_func(2, 3, 10)
        assert output == 35


class TestCompilerWithOptimizer:
    """Tests for compiler working with optimizer."""

    def test_compile_optimized_constants(self):
        """
        Compile a fully constant expression after optimization.
        (2 + 3) * 4 should compile to 20.
        """
        Tracer.reset()
        
        a = Tracer(value=2)
        b = Tracer(value=3)
        c = Tracer(value=4)
        
        ab = a + b  # 5
        result = ab * c  # 20
        
        optimizer = Optimizer()
        optimized = optimizer.optimize(result)
        
        compiler = AthenaCompiler()
        compiled_func = compiler.compile(optimized)
        
        # Should execute correctly (result is 20)
        output = compiled_func()
        assert output == 20

    def test_compile_optimized_with_variables(self):
        """
        Compile after optimizing with variables.
        x + 0 should simplify to x.
        """
        Tracer.reset()
        
        x = Tracer(value=7, name="x")
        result = x + 0
        
        optimizer = Optimizer()
        optimized = optimizer.optimize(result)
        
        compiler = AthenaCompiler()
        compiled_func = compiler.compile(optimized)
        
        output = compiled_func(7)
        assert output == 7


class TestCompilerExecution:
    """Tests for compiled function execution."""

    def test_compiled_function_is_callable(self):
        """Compiled function should be callable."""
        Tracer.reset()
        
        a = Tracer(value=1, name="a")
        b = Tracer(value=2, name="b")
        result = a + b
        
        compiler = AthenaCompiler()
        compiled_func = compiler.compile(result)
        
        assert callable(compiled_func)

    def test_compiled_function_accepts_correct_args(self):
        """Compiled function should accept the right number of arguments."""
        Tracer.reset()
        
        a = Tracer(value=1, name="a")
        b = Tracer(value=2, name="b")
        c = Tracer(value=3, name="c")
        result = a + b + c
        
        compiler = AthenaCompiler()
        compiled_func = compiler.compile(result)
        
        # Should accept 3 arguments
        import inspect
        sig = inspect.signature(compiled_func)
        assert len(sig.parameters) == 3

    def test_compiled_function_returns_correct_type(self):
        """Compiled function should return a number."""
        Tracer.reset()
        
        a = Tracer(value=5, name="a")
        result = a * 2
        
        compiler = AthenaCompiler()
        compiled_func = compiler.compile(result)
        
        output = compiled_func(5)
        assert isinstance(output, (int, float))


class TestCompilerCSE:
    """Tests for CSE in compiled code."""

    def test_cse_reduces_computations(self):
        """
        (a+b) + (a+b) should only compute a+b once in the generated code.
        This is verified by checking the IR or by correct results.
        """
        Tracer.reset()
        
        a = Tracer(value=3, name="a")
        b = Tracer(value=4, name="b")
        
        ab1 = a + b
        ab2 = a + b  # Should share VR
        result = ab1 + ab2
        
        compiler = AthenaCompiler()
        compiled_func = compiler.compile(result)
        
        output = compiled_func(3, 4)
        assert output == 14  # (3+4) + (3+4) = 14

    def test_cse_with_complex_graph(self):
        """
        Complex graph with multiple CSE opportunities.
        """
        Tracer.reset()
        
        x = Tracer(value=2, name="x")
        y = Tracer(value=3, name="y")
        
        # (x+y) * (x+y) + (x+y)
        xy = x + y
        xy_sq = xy * xy
        result = xy_sq + xy
        
        compiler = AthenaCompiler()
        compiled_func = compiler.compile(result)
        
        output = compiled_func(2, 3)
        # (2+3)*(2+3) + (2+3) = 25 + 5 = 30
        assert output == 30

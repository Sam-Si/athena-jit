"""
API Module - Public Interface for Athena JIT

Provides the @jit and @vmap decorators for JIT compilation.
"""

import numpy as np
from athena.tracer import Tracer
from athena.optimizer import Optimizer
from athena.compiler import AthenaCompiler


# Global threshold: minimum number of operations to trigger JIT compilation.
# Below this threshold, raw Python is faster due to JIT call overhead.
# Default of 5 means: a+b (1 op) uses Python; (a+b)*(a+b)+(a+b) (3 ops) uses Python;
# Complex expressions with 5+ operations get JIT compiled.
JIT_COMPLEXITY_THRESHOLD = 5


def jit(func=None, complexity_threshold=None, buffer_mode=False, simd=False):
    """
    Decorator to JIT-compile a Python function using Athena.
    
    Selective JIT Compilation:
        Only JIT compiles when the expression complexity (number of operations)
        meets or exceeds the threshold. Simple operations like `a + b` use
        raw Python which is already fast. Complex expressions with CSE
        opportunities benefit from JIT.
    
    Buffer Mode:
        When buffer_mode=True, the function operates directly on NumPy array
        buffers, eliminating ctypes call overhead. The compiled function
        processes entire arrays in a single call with a tight loop.
    
    SIMD Vectorization:
        When buffer_mode=True and simd=True, the function uses SIMD (Single
        Instruction, Multiple Data) vectorization for ~4x speedup on modern
        CPUs with AVX. Processes 4 doubles per CPU instruction instead of 1.
    
    Args:
        complexity_threshold: Minimum operations to trigger JIT.
                              Default: 5 (from JIT_COMPLEXITY_THRESHOLD)
        buffer_mode: If True, compile for NumPy buffer-direct execution.
                     Function should be called with NumPy arrays.
        simd: If True (with buffer_mode=True), use SIMD vectorization.
              Requires buffer_mode=True. Default: False.
    
    Usage:
        @jit
        def simple(a, b):
            return a + b  # Uses raw Python (1 operation < 5)
        
        @jit(complexity_threshold=3)
        def complex(a, b):
            ab = a + b
            return ab * ab + ab  # JIT compiled (3 operations >= 3)
        
        @jit(buffer_mode=True)
        def array_add(a, b):
            return a + b  # Operates on entire arrays via buffer protocol
        
        @jit(buffer_mode=True, simd=True)
        def fast_add(a, b):
            return a + b  # SIMD-accelerated: ~4x faster than scalar buffer-mode
    """
    if complexity_threshold is None:
        complexity_threshold = JIT_COMPLEXITY_THRESHOLD
    
    def decorator(func):
        # On first call, decide once: JIT or raw Python?
        _cached_result = None
        
        def wrapper(*args, **kwargs):
            nonlocal _cached_result
            
            if _cached_result is None:
                # First call: trace and decide
                Tracer.reset()
                
                # For buffer mode, trace with scalar values to get expression
                # (NumPy arrays will be processed element-wise by compiled loop)
                tracers = []
                for i, arg in enumerate(args):
                    # If buffer mode, use first element or 1.0 as trace value
                    if buffer_mode and hasattr(arg, '__len__') and hasattr(arg, '__getitem__'):
                        # Handle multi-dimensional arrays: get first scalar element
                        try:
                            trace_val = float(np.asarray(arg).flat[0]) if arg.size > 0 else 1.0
                        except (AttributeError, IndexError, ValueError):
                            trace_val = 1.0
                    else:
                        trace_val = arg
                    tracers.append(Tracer(value=trace_val, name=f"arg{i}"))
                
                result_tracer = func(*tracers)
                op_count = Tracer.trace_count()
                
                if buffer_mode:
                    # Buffer mode: always compile for array operations
                    optimizer = Optimizer()
                    optimized = optimizer.optimize(result_tracer)
                    compiler = AthenaCompiler()
                    
                    # Use SIMD if requested
                    if simd:
                        compiled = compiler.compile_simd_buffer_mode(optimized, simd_width=4)
                    else:
                        compiled = compiler.compile_buffer_mode(optimized)
                    
                    def buffer_wrapper(*a, **kw):
                        # Create output array with same shape as first input
                        first_arg = a[0]
                        original_shape = np.asarray(first_arg).shape
                        total_size = np.asarray(first_arg).size
                        
                        # Convert all inputs to float64 1D arrays for buffer processing
                        float_args = [np.asarray(arg, dtype=np.float64).ravel() for arg in a]
                        
                        # Create flat output array
                        out = np.empty(total_size, dtype=np.float64)
                        
                        # Call compiled buffer function (operates on flat arrays)
                        compiled(*float_args, out, total_size)
                        
                        # Reshape to original shape
                        return out.reshape(original_shape)
                    
                    buffer_wrapper.__name__ = func.__name__
                    buffer_wrapper.__doc__ = func.__doc__
                    _cached_result = buffer_wrapper
                
                elif op_count >= complexity_threshold:
                    # Complex: JIT compile, return compiled function
                    optimizer = Optimizer()
                    optimized = optimizer.optimize(result_tracer)
                    compiler = AthenaCompiler()
                    compiled = compiler.compile(optimized)
                    
                    def jit_wrapper(*a, **kw):
                        return compiled(*a)
                    jit_wrapper.__name__ = func.__name__
                    jit_wrapper.__doc__ = func.__doc__
                    _cached_result = jit_wrapper
                else:
                    # Simple: RETURN RAW FUNCTION DIRECTLY - zero overhead!
                    _cached_result = func
            
            return _cached_result(*args, **kwargs)
        
        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        
        return wrapper
    
    if func is not None:
        return decorator(func)
    return decorator


def vmap(func):
    """
    Vectorizing map: transforms a function that operates on scalars 
    into a function that operates on vectors.
    
    This is a placeholder for future vectorization support.
    Currently returns the function unchanged.
    """
    def wrapper(*args, **kwargs):
        # For now, just call the function directly
        # Future: vectorize over array inputs
        return func(*args, **kwargs)
    
    wrapper.__name__ = getattr(func, '__name__', 'vmap_func')
    return wrapper

# Athena-JIT

Athena-JIT is a Minimum Viable Repository (MVR) for a JIT compiler mimicking a subset of JAX. It uses Python as the "brain" (for tracing and orchestration) and LLVM as the "brawn" (for high-performance machine code generation).

The primary objective is to demonstrate the power of JIT compilation by bridging high-level Python abstractions with low-level CPU optimizations.

## Core Features
- **Tracing (JAX-style):** Capturing Python arithmetic operations into a computational graph.
- **Common Subexpression Elimination (CSE):** Recognizing duplicate expressions like `(a+b) + (a+b)` and computing them only once.
- **Constant Folding:** Evaluating arithmetic on constants at compile-time.
- **Algebraic Simplification:** Applying rules like `x + 0 = x` or `x * 0 = 0` to simplify the graph.
- **LLVM Compilation:** Translating the optimized graph into native machine code via MCJIT.
- **SIMD Vectorization:** High-performance array processing using CPU vector instructions (AVX/SSE).

## Key Components
- `athena/api.py`: The `@jit` and `vmap` interfaces.
- `athena/tracer.py`: JAX-style tracers with CSE and virtual register management.
- `athena/optimizer.py`: Optimization passes (constant folding, simplification).
- `athena/compiler.py`: LLVM IR generation (scalar and SIMD) and MCJIT execution.

## Getting Started

### Prerequisites
- Python 3.10+
- `llvmlite` (LLVM 11-19 supported)
- `numpy`

### Installation
1. Install dependencies:
   ```bash
   pip install llvmlite numpy pytest
   ```
2. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

## Usage

### Basic JIT Compilation
```python
from athena.api import jit

@jit
def compute(a, b, c):
    # CSE will ensure (a+b) is computed only once
    ab = a + b
    return ab * ab + c

print(compute(2, 3, 10)) # (2+3)^2 + 10 = 35.0
```

### SIMD Array Processing
Athena supports high-performance array processing using SIMD instructions.
```python
import numpy as np
from athena.api import jit

@jit(buffer_mode=True, simd=True)
def fast_add(a, b):
    return a + b

a = np.random.randn(1000000)
b = np.random.randn(1000000)
result = fast_add(a, b) # Processes 4-8 elements per instruction!
```

## Running Tests
Athena uses `pytest` for verification. The suite includes 59 tests covering all core components.
```bash
pytest
```

## Architecture
1. **Tracing:** Python operations are overloaded to record into a `Tracer` graph.
2. **Optimization:** The `Optimizer` walks the graph to simplify it.
3. **Codegen:** `AthenaCompiler` translates the graph into LLVM IR.
4. **Execution:** `llvmlite` compiles the IR to machine code and returns a callable function via `ctypes`.

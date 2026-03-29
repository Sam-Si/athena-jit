# Athena-JIT

Athena-JIT is a Minimum Viable Repository (MVR) for a JIT compiler mimicking a subset of JAX. It uses Python as the "brain" (for tracing and orchestration) and LLVM as the "brawn" (for high-performance machine code generation).

The name **Athena** was chosen to represent the goddess of wisdom and strategy, reflecting our approach of using Python for high-level "strategic" tracing and LLVM for "wise" and efficient machine-code execution.

## Goal
The primary objective is to demonstrate the power of JIT compilation by bridging the gap between interpreted Python and native machine code performance, focusing on:
- **Tracing:** Capturing Python logic into a linear execution plan.
- **LLVM Compilation:** Translating the plan into optimized CPU instructions.
- **Vectorization (vmap):** Leveraging SIMD for array processing.
- **Operator Fusion:** Minimizing memory overhead by combining operations.

## Key Components
- `athena/api.py`: The `@jit` and `vmap` interfaces.
- `athena/compiler.py`: LLVM ORCv2/MCJIT logic.
- `athena/tracer.py`: JAX-style tracer objects for expression recording.
- `athena/optimizer.py`: Constant folding and fusion passes.

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Run benchmarks: `python benchmarks/compare_jax.py`
3. Run tests: `pytest`

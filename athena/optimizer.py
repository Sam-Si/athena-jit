"""
Optimizer Module - Optimization Passes for Athena JIT

Performs optimization passes on traced computational graphs:
- Constant Folding: Evaluate constant expressions at compile time
- Algebraic Simplifications: x+0=x, x*1=x, x*0=0
- Operator Fusion: Combine compatible operations
- CSE Preservation: Maintain virtual register reuse from tracing
"""

from typing import Any, Optional, Set
from athena.tracer import Tracer


class Optimizer:
    """
    Performs optimization passes on the traced execution plan.
    
    Optimizations include:
    - Constant folding: (2 + 3) -> 5
    - Algebraic simplifications: x + 0 -> x, x * 1 -> x
    - Preserves CSE from tracer
    """
    
    def __init__(self):
        """Initialize the optimizer."""
        self._folded_constants: Set[int] = set()  # Track which VRs were folded
    
    def optimize(self, tracer: Tracer) -> Tracer:
        """
        Optimize a tracer graph.
        
        Args:
            tracer: The root tracer of the computation graph
        
        Returns:
            An optimized tracer (may be the same or a simplified version)
        """
        # Reset state for fresh optimization
        self._folded_constants = set()
        
        # Apply optimization passes
        optimized = self._constant_fold(tracer)
        optimized = self._algebraic_simplify(optimized)
        
        return optimized
    
    def _constant_fold(self, tracer: Tracer) -> Tracer:
        """
        Recursively fold constant expressions.
        
        If both operands of an operation are pure constants (no name, no operation),
        compute the result at compile time.
        Variables (with names) are never folded.
        """
        # Base case: input tracer (no operation)
        if tracer.op is None:
            return tracer
        
        # First, recursively optimize children
        optimized_parents = [self._constant_fold(p) for p in tracer.parents]
        
        # Check if all parents are PURE constants:
        # - No operation (op is None)
        # - No name (not a variable)
        all_pure_constants = all(
            p.op is None and p.name is None 
            for p in optimized_parents
        )
        
        if all_pure_constants and tracer.op in ('add', 'mul'):
            # Fold: compute the result
            if tracer.op == 'add':
                folded_value = optimized_parents[0].value + optimized_parents[1].value
            else:  # mul
                folded_value = optimized_parents[0].value * optimized_parents[1].value
            
            # Return a constant tracer (no name = pure constant)
            return Tracer(value=folded_value)
        
        # Not foldable, return with optimized parents
        # But we need to preserve the original tracer's VR for CSE
        if optimized_parents != tracer.parents:
            # Create new tracer with optimized parents
            return Tracer(
                value=tracer.value,
                op=tracer.op,
                parents=optimized_parents,
                virtual_register=tracer.virtual_register
            )
        
        return tracer
    
    def _algebraic_simplify(self, tracer: Tracer) -> Tracer:
        """
        Apply algebraic simplifications:
        - x + 0 = x (only if 0 is a pure constant, not a variable)
        - x * 1 = x (only if 1 is a pure constant, not a variable)
        - x * 0 = 0 (only if 0 is a pure constant, not a variable)
        
        Pure constants have no name (not variables).
        """
        if tracer.op is None:
            return tracer
        
        # Recursively simplify children first
        simplified_parents = [self._algebraic_simplify(p) for p in tracer.parents]
        
        if tracer.op == 'add':
            # x + 0 = x or 0 + x = x (only for pure constants)
            for i, parent in enumerate(simplified_parents):
                if parent.op is None and parent.name is None and parent.value == 0:
                    other = simplified_parents[1 - i]
                    return other
        
        elif tracer.op == 'mul':
            # x * 0 = 0 or 0 * x = 0 (only for pure constants)
            for parent in simplified_parents:
                if parent.op is None and parent.name is None and parent.value == 0:
                    return Tracer(value=0)
            
            # x * 1 = x or 1 * x = x (only for pure constants)
            for i, parent in enumerate(simplified_parents):
                if parent.op is None and parent.name is None and parent.value == 1:
                    other = simplified_parents[1 - i]
                    return other
        
        # No simplification applied, return with potentially simplified parents
        if simplified_parents != tracer.parents:
            return Tracer(
                value=tracer.value,
                op=tracer.op,
                parents=simplified_parents,
                virtual_register=tracer.virtual_register
            )
        
        return tracer
    
    def _fuse_operations(self, tracer: Tracer) -> Tracer:
        """
        Attempt to fuse compatible operations.
        
        For example: (a + b) + c could potentially be fused.
        This is a placeholder for more advanced fusion.
        """
        # For now, just return as-is
        # Future: implement actual operation fusion for SIMD/vectorization
        return tracer

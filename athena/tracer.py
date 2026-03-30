"""
Tracer Module - The "Brain" of Athena JIT

This module provides JAX-style tracer objects that record Python operations
into a computational graph. Key features:
- Common Subexpression Elimination (CSE): (a+b) + (a+b) reuses the a+b node
- Virtual Register Management: Each unique operation gets a virtual register
- Graph Structure: Full computational graph with parents/operands tracking
"""

from typing import Any, Optional, List, Dict, Tuple, Set


class Tracer:
    """
    JAX-style Tracer objects to record operations for JIT compilation.
    
    Each Tracer represents a value in the computation graph. When arithmetic
    operations are performed on Tracers, they record the operation and create
    new Tracers that form the graph structure.
    
    Key features:
    - CSE: Identical subexpressions share the same virtual register
    - Virtual registers: Unique IDs for each operation (inputs and computed values)
    - Graph tracking: Parents/operands and operation type stored for each node
    """
    
    # Class-level counter for virtual registers
    _vr_counter = 0
    
    # Class-level registry for CSE: maps (op, left_vr, right_vr) -> Tracer
    # This enables detecting duplicate operations
    _cse_registry: Dict[Tuple[str, int, int], 'Tracer'] = {}
    
    # Class-level registry for input tracers by name
    _input_registry: Dict[str, 'Tracer'] = {}
    
    # Class-level registry for constant values (value-based CSE)
    _constant_registry: Dict[Any, 'Tracer'] = {}
    
    # Class-level list of all unique operations (for trace_count)
    _all_operations: List['Tracer'] = []
    
    def __init__(self, value: Any = None, name: Optional[str] = None,
                 op: Optional[str] = None, parents: Optional[List['Tracer']] = None,
                 virtual_register: Optional[int] = None):
        """
        Initialize a Tracer.
        
        Args:
            value: The concrete value (for eager evaluation during tracing)
            name: Optional name for input variables
            op: Operation that created this tracer ('add', 'mul', or None for inputs)
            parents: List of parent tracers (operands)
            virtual_register: Pre-assigned virtual register (used by CSE)
        """
        self.value = value
        self.name = name
        self.op = op
        self.parents = parents or []
        
        # Assign virtual register
        if virtual_register is not None:
            self.virtual_register = virtual_register
        else:
            # For inputs with names, check input registry for CSE
            if name is not None and name in Tracer._input_registry:
                self.virtual_register = Tracer._input_registry[name].virtual_register
            # For constants (no name, no parents), check constant registry for CSE
            elif name is None and (parents is None or len(parents) == 0) and value is not None:
                if value in Tracer._constant_registry:
                    self.virtual_register = Tracer._constant_registry[value].virtual_register
                else:
                    self.virtual_register = Tracer._get_next_vr()
                    Tracer._constant_registry[value] = self
            else:
                self.virtual_register = Tracer._get_next_vr()
                if name is not None:
                    Tracer._input_registry[name] = self
        
        # For operation results, register for CSE
        if op is not None and parents:
            # Create a canonical key for CSE lookup
            parent_vrs = tuple(sorted([p.virtual_register for p in parents]))
            cse_key = (op,) + parent_vrs
            
            # Check if this exact operation already exists
            if cse_key in Tracer._cse_registry:
                # Reuse the existing tracer's virtual register
                existing = Tracer._cse_registry[cse_key]
                self.virtual_register = existing.virtual_register
            else:
                # Register this new operation
                Tracer._cse_registry[cse_key] = self
                Tracer._all_operations.append(self)
    
    @classmethod
    def _get_next_vr(cls) -> int:
        """Get the next available virtual register number."""
        cls._vr_counter += 1
        return cls._vr_counter
    
    @classmethod
    def reset(cls):
        """
        Reset all class-level state.
        
        Useful for testing or starting a new trace.
        """
        cls._vr_counter = 0
        cls._cse_registry.clear()
        cls._input_registry.clear()
        cls._constant_registry.clear()
        cls._all_operations.clear()
    
    @classmethod
    def trace_count(cls) -> int:
        """Return the count of unique operations traced."""
        return len(cls._all_operations)
    
    @classmethod
    def get_all_operations(cls) -> List['Tracer']:
        """Return list of all unique operations traced."""
        return list(cls._all_operations)
    
    def get_graph(self) -> Dict[str, Any]:
        """
        Reconstruct and return the computational graph.
        
        Returns a dictionary representation of the graph rooted at this node.
        """
        def build_graph(node: 'Tracer', visited: Set[int]) -> Dict[str, Any]:
            vr = node.virtual_register
            if vr in visited:
                return {'virtual_register': vr, 'name': node.name, 'ref': True}
            
            visited.add(vr)
            
            result = {
                'virtual_register': vr,
                'name': node.name,
                'value': node.value,
                'op': node.op,
            }
            
            if node.parents:
                result['operands'] = [
                    build_graph(p, visited.copy()) for p in node.parents
                ]
            
            return result
        
        return build_graph(self, set())
    
    def _binary_op(self, other: Any, op_name: str) -> 'Tracer':
        """
        Perform a binary operation, handling both Tracer and constant operands.
        
        Args:
            other: The other operand (Tracer or constant)
            op_name: Operation name ('add' or 'mul')
        
        Returns:
            A new Tracer representing the result
        """
        # Convert constants to Tracers for uniform handling
        if not isinstance(other, Tracer):
            other = Tracer(value=other)
        
        # Compute the result value eagerly (for validation during tracing)
        if op_name == 'add':
            result_value = self.value + other.value
        elif op_name == 'mul':
            result_value = self.value * other.value
        else:
            raise ValueError(f"Unknown operation: {op_name}")
        
        # Create the result tracer (CSE happens in __init__)
        return Tracer(
            value=result_value,
            op=op_name,
            parents=[self, other]
        )
    
    def __add__(self, other: Any) -> 'Tracer':
        """Addition operator: self + other"""
        return self._binary_op(other, 'add')
    
    def __radd__(self, other: Any) -> 'Tracer':
        """Reverse addition: other + self (for constants on left)"""
        # Convert constant to Tracer and add
        return Tracer(value=other).__add__(self)
    
    def __mul__(self, other: Any) -> 'Tracer':
        """Multiplication operator: self * other"""
        return self._binary_op(other, 'mul')
    
    def __rmul__(self, other: Any) -> 'Tracer':
        """Reverse multiplication: other * self (for constants on left)"""
        return Tracer(value=other).__mul__(self)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        if self.name:
            return f"Tracer({self.name}, vr={self.virtual_register}, val={self.value})"
        elif self.op:
            return f"Tracer({self.op}, vr={self.virtual_register}, val={self.value})"
        else:
            return f"Tracer(vr={self.virtual_register}, val={self.value})"


"""
Compiler Module - LLVM IR Generation and Machine Code Compilation

The "Brawn" of Athena JIT: translates traced computational graphs
into optimized LLVM IR and compiles to native machine code.

Key features:
- Generates LLVM IR from Tracer graphs
- Supports add, mul operations
- Compiles to native code via MCJIT
- Returns callable Python functions
"""

import llvmlite.binding as llvm
from llvmlite import ir
from typing import Callable, List, Dict, Any
from athena.tracer import Tracer


class AthenaCompiler:
    """
    Compiles Athena Tracer graphs to native machine code via LLVM.
    
    Workflow:
    1. Take a Tracer (root of computation graph)
    2. Generate LLVM IR module with a function
    3. Compile to machine code
    4. Return a callable Python wrapper
    """
    
    def __init__(self):
        """Initialize LLVM and create target machine."""
        # Initialize native targets (modern llvmlite handles core init automatically)
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        
        # Create target machine for native compilation
        self.target = llvm.Target.from_default_triple()
        self.target_machine = self.target.create_target_machine()
    
    def compile(self, tracer: Tracer) -> Callable:
        """
        Compile a Tracer graph to a callable function.
        
        Args:
            tracer: Root of the computation graph
        
        Returns:
            A callable Python function that executes the compiled code
        """
        # 1. Generate LLVM IR
        ir_module, func_name, arg_names = self._generate_ir(tracer)
        
        # 2. Compile to machine code
        compiled_func = self._compile_ir(ir_module, func_name, arg_names)
        
        return compiled_func
    
    def compile_buffer_mode(self, tracer: Tracer) -> Callable:
        """
        Compile a Tracer graph to a buffer-mode function.
        
        Buffer-mode functions operate directly on NumPy array buffers:
        - Takes pointers to input arrays (double*)
        - Takes pointer to output array (double*)
        - Takes array length (size_t)
        - Loops over all elements, applying the traced expression
        
        This eliminates ctypes call overhead by processing entire arrays
        in a single compiled function call.
        
        Args:
            tracer: Root of the computation graph (traced with scalar values)
        
        Returns:
            A callable function: func(ptr0, ptr1, ..., out_ptr, n)
        """
        # 1. Generate buffer-mode LLVM IR
        ir_module, func_name, arg_names = self._generate_buffer_ir(tracer)
        
        # 2. Compile to machine code
        compiled_func = self._compile_buffer_ir(ir_module, func_name, arg_names)
        
        return compiled_func
    
    def _generate_buffer_ir(self, tracer: Tracer) -> tuple:
        """
        Generate LLVM IR for buffer-mode execution.
        
        Creates a function: void func(double* arg0, double* arg1, ..., double* out, size_t n)
        The function loops from 0 to n, applying the traced expression element-wise.
        
        Returns:
            (ir_module, function_name, arg_names)
        """
        # Collect input variables
        inputs = self._collect_inputs(tracer)
        arg_names = [t.name for t in inputs]
        num_args = len(arg_names)
        
        # Create LLVM IR module
        module = ir.Module(name="athena_buffer_module")
        
        # Types
        double_type = ir.DoubleType()
        double_ptr_type = ir.PointerType(double_type)
        size_type = ir.IntType(64)  # size_t
        
        # Function signature: void func(double* arg0, double* arg1, ..., double* out, size_t n)
        # Arguments: input pointers + output pointer + length
        arg_types = [double_ptr_type] * num_args + [double_ptr_type, size_type]
        func_type = ir.FunctionType(ir.VoidType(), arg_types)
        
        func_name = "athena_buffer_func"
        function = ir.Function(module, func_type, name=func_name)
        
        # Name arguments
        for i, (t, arg) in enumerate(zip(inputs, function.args[:num_args])):
            if t.name:
                arg.name = f"athena_{t.name}_ptr"
            else:
                arg.name = f"athena_arg{i}_ptr"
        function.args[num_args].name = "athena_out_ptr"
        function.args[num_args + 1].name = "athena_n"
        
        # Create entry block
        builder = ir.IRBuilder()
        entry_block = function.append_basic_block(name="entry")
        builder.position_at_end(entry_block)
        
        # Allocate index counter
        i_ptr = builder.alloca(size_type, name="i")
        builder.store(ir.Constant(size_type, 0), i_ptr)
        
        # Create loop blocks: cond -> body -> cond or end
        cond_block = function.append_basic_block(name="cond")
        body_block = function.append_basic_block(name="body")
        end_block = function.append_basic_block(name="end")
        
        builder.branch(cond_block)
        
        # Condition block: check i < n
        builder.position_at_end(cond_block)
        i = builder.load(i_ptr, name="i")
        n = function.args[num_args + 1]
        cond = builder.icmp_unsigned('<', i, n, name="cond")
        builder.cbranch(cond, body_block, end_block)
        
        # Body block: load inputs, compute, store output
        builder.position_at_end(body_block)
        
        # Load input elements at index i
        vr_to_llvm: Dict[int, ir.Value] = {}
        
        for idx, (t, arg_ptr) in enumerate(zip(inputs, function.args[:num_args])):
            # Get pointer to element: arg_ptr + i
            elem_ptr = builder.gep(arg_ptr, [i], name=f"elem_ptr_{idx}")
            elem_val = builder.load(elem_ptr, name=f"elem_{idx}")
            vr_to_llvm[t.virtual_register] = elem_val
        
        # Generate computation graph (same as scalar, but with loaded values)
        ordered_nodes = self._topological_sort(tracer)
        
        for node in ordered_nodes:
            if node.op is None:
                continue
            
            operand_values = []
            for p in node.parents:
                if p.virtual_register in vr_to_llvm:
                    operand_values.append(vr_to_llvm[p.virtual_register])
                else:
                    operand_values.append(ir.Constant(double_type, p.value))
            
            if node.op == 'add':
                result = builder.fadd(operand_values[0], operand_values[1], 
                                      name=f"athena_add_{node.virtual_register}")
            elif node.op == 'mul':
                result = builder.fmul(operand_values[0], operand_values[1],
                                      name=f"athena_mul_{node.virtual_register}")
            else:
                raise ValueError(f"Unknown operation: {node.op}")
            
            vr_to_llvm[node.virtual_register] = result
        
        # Store result to output array
        if tracer.virtual_register in vr_to_llvm:
            result_val = vr_to_llvm[tracer.virtual_register]
        else:
            result_val = ir.Constant(double_type, tracer.value)
        
        out_ptr = function.args[num_args]
        out_elem_ptr = builder.gep(out_ptr, [i], name="out_elem_ptr")
        builder.store(result_val, out_elem_ptr)
        
        # Increment i
        i_next = builder.add(i, ir.Constant(size_type, 1), name="i_next")
        builder.store(i_next, i_ptr)
        
        # Back to condition
        builder.branch(cond_block)
        
        # End block
        builder.position_at_end(end_block)
        builder.ret_void()
        
        return module, func_name, arg_names
    
    def _generate_simd_ir(self, tracer: Tracer, simd_width: int = 4) -> tuple:
        """
        Generate LLVM IR for SIMD buffer-mode execution.
        
        Creates a function using vector types:
            void func(double* arg0, ..., double* out, size_t n)
        
        The function processes SIMD_WIDTH elements per iteration using
        vectorized loads, stores, and arithmetic operations.
        Handles remainder elements with a scalar loop.
        """
        # Collect input variables
        inputs = self._collect_inputs(tracer)
        arg_names = [t.name for t in inputs]
        num_args = len(arg_names)
        
        # Create LLVM IR module
        module = ir.Module(name="athena_simd_module")
        
        # Types
        double_type = ir.DoubleType()
        vector_type = ir.VectorType(double_type, simd_width)
        vector_ptr_type = ir.PointerType(vector_type)
        size_type = ir.IntType(64)
        double_ptr_type = ir.PointerType(double_type)
        
        # Function signature
        arg_types = [double_ptr_type] * num_args + [double_ptr_type, size_type]
        func_type = ir.FunctionType(ir.VoidType(), arg_types)
        
        func_name = "athena_simd_buffer_func"
        function = ir.Function(module, func_type, name=func_name)
        
        # Name arguments
        for i, (t, arg) in enumerate(zip(inputs, function.args[:num_args])):
            arg.name = f"athena_{t.name}_ptr" if t.name else f"athena_arg{i}_ptr"
        function.args[num_args].name = "athena_out_ptr"
        function.args[num_args + 1].name = "athena_n"
        
        # Create blocks
        entry_block = function.append_basic_block(name="entry")
        vec_cond_block = function.append_basic_block(name="vec_cond")
        vec_body_block = function.append_basic_block(name="vec_body")
        scalar_cond_block = function.append_basic_block(name="scalar_cond")
        scalar_body_block = function.append_basic_block(name="scalar_body")
        end_block = function.append_basic_block(name="end")
        
        # === Entry Block ===
        builder = ir.IRBuilder(entry_block)
        n = function.args[num_args + 1]
        simd_width_const = ir.Constant(size_type, simd_width)
        
        n_vec = builder.udiv(n, simd_width_const, name="n_vec")
        
        # Vector loop counter
        vec_i_ptr = builder.alloca(size_type, name="vec_i")
        builder.store(ir.Constant(size_type, 0), vec_i_ptr)
        
        # Scalar loop counter start index: n_vec * simd_width
        scalar_i_ptr = builder.alloca(size_type, name="scalar_i")
        start_idx = builder.mul(n_vec, simd_width_const, name="start_idx")
        builder.store(start_idx, scalar_i_ptr)
        
        builder.branch(vec_cond_block)
        
        # === Vector Condition Block ===
        builder.position_at_end(vec_cond_block)
        vec_i = builder.load(vec_i_ptr, name="vec_i")
        vec_cond = builder.icmp_unsigned('<', vec_i, n_vec, name="vec_cond")
        builder.cbranch(vec_cond, vec_body_block, scalar_cond_block)
        
        # === Vector Body Block ===
        builder.position_at_end(vec_body_block)
        
        vr_to_llvm: Dict[int, ir.Value] = {}
        for idx, (t, arg_ptr) in enumerate(zip(inputs, function.args[:num_args])):
            # Calculate offset: vec_i * simd_width
            offset = builder.mul(vec_i, simd_width_const, name=f"offset_{idx}")
            elem_ptr = builder.gep(arg_ptr, [offset], name=f"vec_elem_ptr_{idx}")
            vec_ptr = builder.bitcast(elem_ptr, vector_ptr_type, name=f"vec_ptr_{idx}")
            vec_val = builder.load(vec_ptr, name=f"vec_{idx}")
            vr_to_llvm[t.virtual_register] = vec_val
            
        ordered_nodes = self._topological_sort(tracer)
        for node in ordered_nodes:
            if node.op is None: continue
            
            operand_values = []
            for p in node.parents:
                if p.virtual_register in vr_to_llvm:
                    operand_values.append(vr_to_llvm[p.virtual_register])
                else:
                    scalar_const = ir.Constant(double_type, p.value)
                    operand_values.append(ir.Constant(vector_type, [scalar_const] * simd_width))
            
            if node.op == 'add':
                res = builder.fadd(operand_values[0], operand_values[1], name=f"vadd_{node.virtual_register}")
            elif node.op == 'mul':
                res = builder.fmul(operand_values[0], operand_values[1], name=f"vmul_{node.virtual_register}")
            else:
                raise ValueError(f"Unknown op: {node.op}")
            vr_to_llvm[node.virtual_register] = res
            
        # Store vector result
        out_ptr = function.args[num_args]
        out_offset = builder.mul(vec_i, simd_width_const, name="out_offset")
        out_elem_ptr = builder.gep(out_ptr, [out_offset], name="out_vec_elem_ptr")
        out_vec_ptr = builder.bitcast(out_elem_ptr, vector_ptr_type, name="out_vec_ptr")
        
        if tracer.virtual_register in vr_to_llvm:
            res_vec = vr_to_llvm[tracer.virtual_register]
        else:
            scalar_const = ir.Constant(double_type, tracer.value)
            res_vec = ir.Constant(vector_type, [scalar_const] * simd_width)
        builder.store(res_vec, out_vec_ptr)
        
        # Increment and loop
        vec_i_next = builder.add(vec_i, ir.Constant(size_type, 1), name="vec_i_next")
        builder.store(vec_i_next, vec_i_ptr)
        builder.branch(vec_cond_block)
        
        # === Scalar Condition Block ===
        builder.position_at_end(scalar_cond_block)
        scalar_i = builder.load(scalar_i_ptr, name="scalar_i")
        scalar_cond = builder.icmp_unsigned('<', scalar_i, n, name="scalar_cond")
        builder.cbranch(scalar_cond, scalar_body_block, end_block)
        
        # === Scalar Body Block ===
        builder.position_at_end(scalar_body_block)
        vr_to_llvm_scalar: Dict[int, ir.Value] = {}
        for idx, (t, arg_ptr) in enumerate(zip(inputs, function.args[:num_args])):
            elem_ptr = builder.gep(arg_ptr, [scalar_i], name=f"elem_ptr_{idx}")
            vr_to_llvm_scalar[t.virtual_register] = builder.load(elem_ptr, name=f"val_{idx}")
            
        for node in ordered_nodes:
            if node.op is None: continue
            operand_values = []
            for p in node.parents:
                if p.virtual_register in vr_to_llvm_scalar:
                    operand_values.append(vr_to_llvm_scalar[p.virtual_register])
                else:
                    operand_values.append(ir.Constant(double_type, p.value))
            
            if node.op == 'add':
                res = builder.fadd(operand_values[0], operand_values[1], name=f"sadd_{node.virtual_register}")
            elif node.op == 'mul':
                res = builder.fmul(operand_values[0], operand_values[1], name=f"smul_{node.virtual_register}")
            else:
                raise ValueError(f"Unknown op: {node.op}")
            vr_to_llvm_scalar[node.virtual_register] = res
            
        # Store scalar result
        out_elem_ptr = builder.gep(out_ptr, [scalar_i], name="out_elem_ptr")
        if tracer.virtual_register in vr_to_llvm_scalar:
            res_val = vr_to_llvm_scalar[tracer.virtual_register]
        else:
            res_val = ir.Constant(double_type, tracer.value)
        builder.store(res_val, out_elem_ptr)
        
        # Increment and loop
        scalar_i_next = builder.add(scalar_i, ir.Constant(size_type, 1), name="scalar_i_next")
        builder.store(scalar_i_next, scalar_i_ptr)
        builder.branch(scalar_cond_block)
        
        # === End Block ===
        builder.position_at_end(end_block)
        builder.ret_void()
        
        return module, func_name, arg_names
    
    def compile_simd_buffer_mode(self, tracer: Tracer, simd_width: int = 4) -> Callable:
        """
        Compile a Tracer graph to a SIMD buffer-mode function.
        
        SIMD buffer-mode functions use vector types for parallel processing:
        - Processes SIMD_WIDTH elements per loop iteration
        - Uses vectorized loads, stores, and arithmetic
        - Handles remainder elements with scalar code
        
        Args:
            tracer: Root of the computation graph
            simd_width: Vector width (default 4 for AVX)
        
        Returns:
            A callable function: func(ptr0, ptr1, ..., out_ptr, n)
        """
        import ctypes
        
        # Generate SIMD IR
        ir_module, func_name, arg_names = self._generate_simd_ir(tracer, simd_width)
        
        # Parse and compile
        llvm_ir = str(ir_module)
        llvm_module = llvm.parse_assembly(llvm_ir)
        llvm_module.verify()
        
        engine = llvm.create_mcjit_compiler(llvm_module, self.target_machine)
        engine.finalize_object()
        engine.run_static_constructors()
        
        func_ptr = engine.get_function_address(func_name)
        
        # Create ctypes wrapper
        num_args = len(arg_names)
        arg_types = [ctypes.c_void_p] * num_args + [ctypes.c_void_p, ctypes.c_size_t]
        c_func = ctypes.CFUNCTYPE(None, *arg_types)(func_ptr)
        
        def wrapper(*args):
            """Wrapper that accepts NumPy arrays or pointers."""
            ptrs = []
            n = None
            
            for i, arg in enumerate(args):
                if i < num_args:
                    if hasattr(arg, 'ctypes'):
                        ptrs.append(arg.ctypes.data)
                        if n is None:
                            n = len(arg)
                    else:
                        ptrs.append(arg)
                elif i == num_args:
                    if hasattr(arg, 'ctypes'):
                        ptrs.append(arg.ctypes.data)
                    else:
                        ptrs.append(arg)
                else:
                    n = arg
            
            if n is None:
                raise ValueError("Could not determine array length")
            
            c_func(*ptrs, n)
        
        wrapper._engine = engine
        wrapper._c_func = c_func
        
        return wrapper
    
    def _compile_buffer_ir(self, ir_module: ir.Module, func_name: str, 
                           arg_names: List[str]) -> Callable:
        """
        Compile buffer-mode LLVM IR to a callable Python function.
        
        The compiled function signature is:
            func(ptr0, ptr1, ..., out_ptr, n)
        
        Where ptrs are ctypes pointers and n is the array length.
        """
        import ctypes
        
        # Parse the IR module
        llvm_ir = str(ir_module)
        llvm_module = llvm.parse_assembly(llvm_ir)
        llvm_module.verify()
        
        # Create MCJIT compiler
        engine = llvm.create_mcjit_compiler(llvm_module, self.target_machine)
        
        # Finalize object
        engine.finalize_object()
        engine.run_static_constructors()
        
        # Get the function pointer
        func_ptr = engine.get_function_address(func_name)
        
        # Create ctypes wrapper
        # Function: void func(double*, double*, ..., double*, size_t)
        num_args = len(arg_names)
        
        # Build arg types: double* for each input, double* for output, size_t for n
        arg_types = [ctypes.c_void_p] * num_args + [ctypes.c_void_p, ctypes.c_size_t]
        
        c_func = ctypes.CFUNCTYPE(None, *arg_types)(func_ptr)
        
        def wrapper(*args):
            """
            Wrapper that accepts NumPy arrays or pointers.
            
            Can be called as:
                wrapper(arr0, arr1, out, n)  # with numpy arrays
                wrapper(ptr0, ptr1, out_ptr, n)  # with raw pointers
            """
            # Extract pointers from numpy arrays if needed
            ptrs = []
            n = None
            
            for i, arg in enumerate(args):
                if i < num_args:
                    # Input array
                    if hasattr(arg, 'ctypes'):
                        # NumPy array - get data pointer
                        ptrs.append(arg.ctypes.data)
                        if n is None:
                            n = len(arg)
                    else:
                        # Assume it's already a pointer
                        ptrs.append(arg)
                elif i == num_args:
                    # Output array
                    if hasattr(arg, 'ctypes'):
                        ptrs.append(arg.ctypes.data)
                    else:
                        ptrs.append(arg)
                else:
                    # Length
                    n = arg
            
            if n is None:
                raise ValueError("Could not determine array length")
            
            # Call the compiled function
            c_func(*ptrs, n)
        
        # Keep references to prevent GC
        wrapper._engine = engine
        wrapper._c_func = c_func
        
        return wrapper
    
    def _generate_ir(self, tracer: Tracer) -> tuple:
        """
        Generate LLVM IR module from a Tracer graph.
        
        Returns:
            (ir_module, function_name, arg_names)
        """
        # Collect all unique input variables (tracers with names and no ops)
        inputs = self._collect_inputs(tracer)
        arg_names = [t.name for t in inputs]
        num_args = len(arg_names)
        
        # Create LLVM IR module
        module = ir.Module(name="athena_module")
        
        # Define function signature: double f(double arg0, double arg1, ...)
        double_type = ir.DoubleType()
        arg_types = [double_type] * num_args
        func_type = ir.FunctionType(double_type, arg_types)
        
        func_name = "athena_jit_func"
        function = ir.Function(module, func_type, name=func_name)
        
        # Symbol mapping: Use actual variable names from tracer (athena symbol mapping)
        for i, (t, arg) in enumerate(zip(inputs, function.args)):
            # Use tracer name if available, else use athena_arg prefix
            if t.name:
                arg.name = f"athena_{t.name}"
            else:
                arg.name = f"athena_arg{i}"
        
        # Create entry block
        builder = ir.IRBuilder()
        entry_block = function.append_basic_block(name="entry")
        builder.position_at_end(entry_block)
        
        # Map from virtual register to LLVM value
        vr_to_llvm: Dict[int, ir.Value] = {}
        
        # Load arguments into VR map
        for i, (t, arg) in enumerate(zip(inputs, function.args)):
            vr_to_llvm[t.virtual_register] = arg
        
        # Generate IR for the computation graph (topological order)
        # We need to visit nodes in dependency order
        ordered_nodes = self._topological_sort(tracer)
        
        for node in ordered_nodes:
            if node.op is None:
                # Input node - already handled above
                continue
            
            # Get LLVM values for operands
            # Handle case where operand might be a constant (not in vr_to_llvm)
            operand_values = []
            for p in node.parents:
                if p.virtual_register in vr_to_llvm:
                    operand_values.append(vr_to_llvm[p.virtual_register])
                else:
                    # Constant - create a constant in LLVM
                    operand_values.append(ir.Constant(double_type, p.value))
            
            # Generate the operation with athena prefix (symbol mapping)
            if node.op == 'add':
                result = builder.fadd(operand_values[0], operand_values[1], name=f"athena_add_{node.virtual_register}")
            elif node.op == 'mul':
                result = builder.fmul(operand_values[0], operand_values[1], name=f"athena_mul_{node.virtual_register}")
            else:
                raise ValueError(f"Unknown operation: {node.op}")
            
            vr_to_llvm[node.virtual_register] = result
        
        # Return the final result
        # Handle case where final result is a constant (fully folded)
        if tracer.virtual_register in vr_to_llvm:
            final_value = vr_to_llvm[tracer.virtual_register]
        else:
            # Constant result - create LLVM constant
            final_value = ir.Constant(double_type, tracer.value)
        
        builder.ret(final_value)
        
        return module, func_name, arg_names
    
    def _collect_inputs(self, tracer: Tracer) -> List[Tracer]:
        """
        Collect all input tracers (variables) in the graph.
        
        Returns them in a consistent order (by virtual register).
        """
        inputs = []
        visited = set()
        
        def visit(node: Tracer):
            if node.virtual_register in visited:
                return
            visited.add(node.virtual_register)
            
            if node.op is None and node.name is not None:
                inputs.append(node)
            
            for parent in node.parents:
                visit(parent)
        
        visit(tracer)
        
        # Sort by virtual register for consistent ordering
        inputs.sort(key=lambda t: t.virtual_register)
        return inputs
    
    def _topological_sort(self, tracer: Tracer) -> List[Tracer]:
        """
        Return nodes in topological order (dependencies first).
        """
        result = []
        visited = set()
        
        def visit(node: Tracer):
            if node.virtual_register in visited:
                return
            visited.add(node.virtual_register)
            
            # Visit parents first
            for parent in node.parents:
                visit(parent)
            
            result.append(node)
        
        visit(tracer)
        return result
    
    def _compile_ir(self, ir_module: ir.Module, func_name: str, 
                    arg_names: List[str]) -> Callable:
        """
        Compile LLVM IR module to a callable Python function.
        """
        # Parse the IR module
        llvm_ir = str(ir_module)
        llvm_module = llvm.parse_assembly(llvm_ir)
        llvm_module.verify()
        
        # Create MCJIT compiler
        engine = llvm.create_mcjit_compiler(llvm_module, self.target_machine)
        
        # Finalize object
        engine.finalize_object()
        engine.run_static_constructors()
        
        # Get the function pointer
        func_ptr = engine.get_function_address(func_name)
        
        # Create a ctypes wrapper
        import ctypes
        
        # Function signature: double func(double, double, ...)
        num_args = len(arg_names)
        
        # Keep references to prevent garbage collection
        # We need to keep engine and c_func alive by storing them on the wrapper
        import inspect
        
        if num_args == 0:
            c_func = ctypes.CFUNCTYPE(ctypes.c_double)(func_ptr)
            def wrapper():
                return c_func()
        elif num_args == 1:
            c_func = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)(func_ptr)
            def wrapper(a):
                return c_func(a)
        elif num_args == 2:
            c_func = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)(func_ptr)
            def wrapper(a, b):
                return c_func(a, b)
        elif num_args == 3:
            c_func = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)(func_ptr)
            def wrapper(a, b, c):
                return c_func(a, b, c)
        else:
            # Generic wrapper for more args
            arg_types = [ctypes.c_double] * num_args
            c_func = ctypes.CFUNCTYPE(ctypes.c_double, *arg_types)(func_ptr)
            def wrapper(*args):
                return c_func(*args)
        
        # Store references to prevent garbage collection
        wrapper._engine = engine
        wrapper._c_func = c_func
        
        return wrapper

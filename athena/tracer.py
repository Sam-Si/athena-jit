class Tracer:
    """
    JAX-style Tracer objects to record operations for JIT compilation.
    """
    def __init__(self, value=None, name=None):
        self.value = value
        self.name = name

    def __add__(self, other):
        pass

    def __mul__(self, other):
        pass

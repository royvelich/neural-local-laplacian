# Stub for NeLo's visualization.littlerender_binding.
# The actual module is a C++ binding for a custom renderer
# that is not needed for inference. All calls are no-ops.

def __getattr__(name):
    """Return a no-op callable for any attribute access."""
    return lambda *args, **kwargs: None

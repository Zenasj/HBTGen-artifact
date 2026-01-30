import torch

@torch.compile(backend="aot_eager", fullgraph=True)
def fn(x):
    return x.sin()

x = torch.randn(3, 4)
y = torch.func.vmap(fn)(x)

# ...
# torch/_dynamo/convert_frame.py:178: in _fn
#     return fn(*args, **kwargs)
# torch/_dynamo/convert_frame.py:564: in transform
#     tracer = InstructionTranslator(
# torch/_dynamo/symbolic_convert.py:2396: in __init__
#     self._throw_if_in_functorch()
# torch/_dynamo/symbolic_convert.py:2452: in _throw_if_in_functorch
#     unimplemented(msg)
# torch/_dynamo/exc.py:221: in unimplemented
#     raise Unsupported(msg)
# E   torch._dynamo.exc.Unsupported: torch.func.vmap(fn) requires the function to be inlined by dynamo
# E
# E   Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information
# E
# E
# E   You can suppress this exception and fall back to eager by setting:
# E       import torch._dynamo
# E       torch._dynamo.config.suppress_errors = True
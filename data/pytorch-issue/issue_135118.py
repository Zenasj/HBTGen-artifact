import torch

@torch.compile(backend="eager")
def f():
    _ = torch.autograd.function.FunctionCtx()
    return None
f()
import torch

@torch.jit.script
def foo(x):
    y = torch.neg(x)
    return x - y

print(foo.graph.debug_str())
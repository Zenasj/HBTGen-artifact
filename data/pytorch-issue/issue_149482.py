import torch

@torch.compile(fullgraph=True, backend="eager")
def test():
    x = MySubclass(...)
    x.foo = 42
    return x * x.foo
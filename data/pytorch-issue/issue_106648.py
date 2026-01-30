import torch

@torch.compile()
def foo(x):
    return x.__getitem__([1, 2])

x = torch.rand([5, 5, 5])
# works
foo(x)

@torch.compile()
def foo(x):
    return torch.Tensor.__getitem__([1, 2])

# graph break
foo(x)
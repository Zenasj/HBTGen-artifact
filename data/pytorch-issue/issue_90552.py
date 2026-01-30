import torch

@torch.compile(...)
def f(x):
    leaf = torch.ones(2, requires_grad=True)
    return leaf, leaf * 2

leaf, out = f(torch.ones(2, requires_grad=True))
out.backward()

# This incorrectly prints None. autograd-ing through "out" should have populated `leaf.grad`.
print(leaf.grad)
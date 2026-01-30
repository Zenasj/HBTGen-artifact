import torch

input = torch.rand([2], dtype=torch.float32, requires_grad=True)

def func(input):
    res = torch.amax(input)
    return res

# torch.autograd.functional.jacobian(func, (input, ), strategy='forward-mode', vectorize=True)
# RuntimeError: Could not allocate memory to change Tensor SizesAndStrides!

print(torch.func.jacfwd(func)(input))

import torch
from torch._vmap_internals import vmap

input = torch.randn(2, 2)

def fn(x):
    return x.sum(())

o = vmap(fn)(input)
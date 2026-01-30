py
import torch
import operator

op = operator.iadd

@torch.compile()
def func(x, y):
    op(x, torch.ones(2, 2))
    x.z = y
    op(x, torch.ones(2, 2))
    return x

out = func(torch.ones(2, 2), torch.ones(2, 2))
print(out)
print(out.z)
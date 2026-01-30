import torch, itertools

n = 200
a = torch.randn((n, n, n), device='cuda')

def fn(a):
    t = (a, a + 1, a + 2)
    shape = a.shape
    m = 1
    for dim in range(len(shape)):
        view_shape = [1]*(dim + 1)
        view_shape[dim] = -1
        b = (torch.arange(shape[dim], device=a.device).view(view_shape))
        m = torch.mul(m, b)
    return sum(torch.mul(t1, m) for t1 in t)
    
torch.compile(fn)(a);

import torch

def fn():
    b = torch.ones([10], device='cuda')
    bs = (b for _ in [0])
    return sum(bs)

torch.compile(fn)()
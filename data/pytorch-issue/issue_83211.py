import torch

def fn(x, y):
    return x.reshape(-1, *y.shape[-2:])

torch.jit.script(fn)

import torch

def fn(x, y):
    return x.reshape([-1] + y.shape[-2:])

torch.jit.script(fn)
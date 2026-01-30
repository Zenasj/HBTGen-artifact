import torch.nn as nn

import torch

def fn(x):
    p = torch.nn.Parameter(x)
    return p[0]

if __name__ == '__main__':
    opt_fn = torch.compile(fn, backend="aot_eager")
    opt_fn(torch.rand(3))
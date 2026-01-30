import numpy as np

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, i):
        t1 = torch.fliplr(input=i)
        t2 = torch.resolve_conj(input=t1)
        r1= torch.diagflat(input=t2, offset=100)
        t3 = torch.nn.functional.relu6(inplace=True, input=i)
        return r1

model = Model().to(torch.device("cpu"))
inp = torch.randint(-2147483648, 2147483647, [10, 5, 5, 1, 8, 7], dtype=torch.int32)
ret_eager = model(inp)
ret_exported = torch.compile(model)(inp)

for r1, r2 in zip(ret_eager, ret_exported):
    if not torch.allclose(r1, r2, rtol=1e-2, atol=1e-3, equal_nan=True):
        print("r1: ",r1,"r2: ",r2)
        raise ValueError("Tensors are different.")

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, i):
        t1 = torch.fliplr(input=i)
        t2 = torch.resolve_conj(input=t1)
        r1= torch.diagflat(input=t2, offset=100)
        t3 = torch.nn.functional.relu6(inplace=True, input=i)
        return r1

import torch._inductor
from torch._inductor import config
config.fallback_random = True
model = Model().to(torch.device("cpu"))
inp = torch.randint(-2147483648, 2147483647, [10, 5, 5, 1, 8, 7], dtype=torch.int32)
ret_eager = model(inp.clone())
ret_exported = torch.compile(model)(inp)

torch.testing.assert_allclose(ret_eager, ret_exported)
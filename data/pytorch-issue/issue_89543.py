import torch.nn as nn

#!/usr/bin/env python

import torch

N = 16
# create some observations with missing values
y = torch.randn(N)
y = torch.where(y > .5, y, torch.nan)
x = torch.nn.Parameter(torch.randn(N), requires_grad=True)

# variant 1 correctly produces NaN-free gradient
r = x - y
m = torch.isnan(y)
r = torch.where(m, 0.0, r) **2
loss = r.sum() / m.sum()
loss.backward()
assert not any(torch.isnan(x.grad))  # ✔

# variant 2 incorrectly produces NaN-gradients
x.grad = None
r = (x - y) ** 2
m = torch.isnan(y)
r = torch.where(m, 0.0, r)
loss = r.sum() / m.sum()
loss.backward()
assert not any(torch.isnan(x.grad)) # ✘

import torch

x = torch.nn.Parameter(torch.tensor([1.1, 1.2, 1.3]))
y = torch.tensor([2, 3, 4])
m = torch.tensor([True, True, False])

x.grad = None
r = (x - y) ** 2
r = torch.where(m, r, 0.0)
loss = r.sum() / m.sum()
loss.backward()
print(x.grad)
assert not any(torch.isnan(x.grad))  # ✔

import functorch
from torch import nn, tensor

x = nn.Parameter(tensor([1.1, 1.2, 1.3]))
y = tensor([1, 2, float("nan")])

def f(x):
    r = x - y
    r = r**2
    m = torch.isnan(y)
    r = torch.where(m, 0.0, r)
    return r.sum() / m.sum()

df = functorch.jacfwd(f)
print(df(x))   # [ 0.2000, -1.6000,  0.0000] ✔
df = functorch.jacrev(f)
print(df(x))   # [ 0.2000, -1.6000,     nan] ✘
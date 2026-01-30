py
import torch
from torch.func import jacrev

x = torch.tensor(0.0)

def func(x):
    y = torch.roll(x, 1)
    return y

print(jacrev(func)(x))
# IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)

py
import torch
from torch.autograd.functional import jacobian

x = torch.tensor(0.0)

def func(x):
    y = torch.roll(x, 1)
    return y

print(jacobian(func, x))
# tensor(1.)
import torch


@torch.compile
def f(x, self):
    return x + self


a = torch.randn(3, 4)
b = torch.randn(3, 4)

print(f(a, b))

import torch

from treevalue import FastTreeValue

print('torch version:', torch.__version__)


@torch.compile
def foo(x, y):
    z = x + y
    return z


x = FastTreeValue({'a': torch.randn(3, 4)})
y = FastTreeValue({'a': torch.rand(4)})

print('x:', x)
print('y:', y)

print(foo(x, y))  # this line!!!
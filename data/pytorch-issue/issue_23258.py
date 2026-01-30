from collections import namedtuple

import torch
print(torch.version.__version__)  # 1.9.0; same for 1.7.0 and 1.8.1

A = namedtuple("A", "x y")

def foo(ntpl : A):
    return ntpl.x


a = A(torch.zeros([1,]), torch.ones([1,]) )

print(foo(a))
traced = torch.jit.trace(foo, (a, ))  # AttributeError: 'tuple' object has no attribute 'x'
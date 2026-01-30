import torch.nn as nn

import torch
from collections import namedtuple

def foo(x, y):
    return 2*x + y

NT = namedtuple('a', ['arg0', 'arg1'])
x = NT(torch.rand(3), torch.rand(3))
traced_foo = torch.jit.trace(foo, x)

import torch
from collections import namedtuple

def foo(x, y):
    return 2*x + y

NT = namedtuple('a', ['arg0', 'arg1'])
x = NT(torch.rand(3), torch.rand(3))
traced_foo = torch.jit.trace(foo, tuple(x))

import torch

def foo(x):
    return {'output': 2*x['data0'] + x['data1']}
 

x = {'data0': torch.rand(3), 'data1': torch.rand(3)}
traced_foo = torch.jit.trace(foo, x)

import torch


class Baa(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.foo = Foo()

    def forward(self, a, b):
        return self.foo({'a': a, 'b': b})['a']


class Foo(torch.nn.Module):

    def forward(self, x):
        return {'a': x['a'] * x['b']}


x = (torch.rand(3), torch.rand(3))
model = Baa()
traced_foo = torch.jit.trace(model, x)

from collections import namedtuple

import torch
print(torch.version.__version__)  # 1.9.0; same for 1.7.0 and 1.8.1

A = namedtuple("A", "x y")

def foo(ntpl : A):
    return ntpl.x


a = A(torch.zeros([1,]), torch.ones([1,]) )

print(foo(a))
traced = torch.jit.trace(foo, (a, ))  # AttributeError: 'tuple' object has no attribute 'x'
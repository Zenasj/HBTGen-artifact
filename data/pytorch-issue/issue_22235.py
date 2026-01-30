import sys
import pickle
import copy
import unittest
from collections import OrderedDict

import numpy as np
from numpy.testing import *

import torch
import torch.multiprocessing as mp


def _rebuild_subclass(type_, data, requires_grad, backward_hooks):
    param = type_(data, requires_grad)
    # NB: This line exists only for backwards compatibility; the
    # general expectation is that backward_hooks is an empty
    # OrderedDict.  See Note [Don't serialize hooks]
    param._backward_hooks = backward_hooks

    return param


class TensorSubclass(torch.Tensor):

    def __new__(cls, data=None, requires_grad=False):
        if data is None:
            data = torch.Tensor()
        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        return self

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self.data.clone(), self.requires_grad)
            memo[id(self)] = result
            return result

    def __reduce_ex__(self, proto):
        # See Note [Don't serialize hooks]
        return _rebuild_subclass, (self.__class__, self.data, self.requires_grad, OrderedDict())


class A(TensorSubclass):
    pass


class B(TensorSubclass):
    pass


class C(A, B):
    pass


if __name__ == '__main__':
    a_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.double, requires_grad=True)
    b_tensor = torch.tensor([4.0, 5.0, 6.0], dtype=torch.double, requires_grad=True)
    c_tensor = torch.tensor([7.0, 8.0, 9.0], dtype=torch.double, requires_grad=True)
    d_tensor = torch.ones((4, 3, 2), dtype=torch.double, requires_grad=True)
    a = A(a_tensor, requires_grad=True)
    b = B(b_tensor, requires_grad=True)
    c = C(c_tensor, requires_grad=True)
    d = C(d_tensor, requires_grad=True)
    
    print("Subclass")
    print(type(a))
    print(a.requires_grad)
    print(type(b))
    print(b.requires_grad)
    print(type(c))
    print(c.requires_grad)

    print("Torch ptype propagation")
    print(type(a + b))
    print((a + b).requires_grad)
    print(type(a + c))
    print((a + c).requires_grad)

    print("Indexing [None, elipsis, integer, slice")
    print(type(c[None]))
    print(type(c[...]))
    print(type(c[0]))
    print(type(c[slice(0, 1, 1)]))
    print(type(d[None, ..., 0, slice(0, 1, 1)]))

    print("bool")
    print(a == c)

@time_it
def slow(x, y):
    for i in range(1000):
        x = x + y
    return x
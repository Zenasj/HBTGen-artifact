from typing import Any
import torch

class A:
    def __init__(self, t):
        self.t = t
    @staticmethod
    def f(a: torch.Tensor):
        return A(a + 1)

class B(A):
    def __init__(self, t):
        self.t = t + 10
    @staticmethod
    def f(a: torch.Tensor):
        return A(a + 1)

x = A(torch.tensor([3]))

def fun(x: Any):
    if isinstance(x, A):
        return A.f(x.t)
    else:
        return B.f(x.t)

print(torch.__version__)
sc = torch.jit.script(fun)
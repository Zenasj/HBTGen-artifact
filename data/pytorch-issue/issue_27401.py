import torch
import torch.nn as nn

class A(torch.nn.Module):
    def __init__(self):
        super(A, self).__init__()

    def forward(self, x):
        return x + 3

class B(torch.nn.Module):
    def __init__(self):
        super(B, self).__init__()

    def forward(self, x):
        return {"1": x}

class C(torch.nn.Module):
    __constants__ = ['foo']

    def __init__(self):
        super(C, self).__init__()
        self.foo = torch.nn.Sequential(A(), B())

    def forward(self, x):
        return self.foo(x)

c = C()
print(torch.jit.script(c).graph)
import torch.nn as nn

import torch

def functional(fcn, y0):
    params = fcn.__self__.parameters()  # assuming fcn is a method of `torch.nn.Module`
    return Functional.apply(fcn, y0, *params)  # NO_MEMLEAK_IF: params is an empty list

class Functional(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fcn, y0, *params):
        y = fcn(y0)
        ctx.fcn = fcn  # NO_MEMLEAK_IF: removing this line, but fcn is needed in backward
        return y

class DummyModule(torch.nn.Module):
    def __init__(self, a):
        super().__init__()
        self.a = torch.nn.Parameter(a)
        x0 = torch.ones_like(a)
        xsol = functional(self.forward, x0)  # NO_MEMLEAK_IF: changing this line to self.forward(x0)
        self.xsol = xsol  # NO_MEMLEAK_IF: removing this line or using xsol.detach()

    def forward(self, x):
        return self.a * x

def test_functional():
    a = torch.ones((200000000,), dtype=torch.double, device=torch.device("cuda"))
    model = DummyModule(a)

for i in range(5):
    test_functional()
    torch.cuda.empty_cache()
    print("memory allocated:", float(torch.cuda.memory_allocated() / (1024 ** 2)), "MiB")

def functional(fcn, y0):
    params = [fcn.__self__.a]
    return Functional.apply(fcn, y0, *params)  # NO_MEMLEAK_IF: params is an empty list

class DummyModule(object):
    def __init__(self, a):
        self.a = a
        x0 = torch.ones_like(a)
        xsol = functional(self.function4functional, x0)  # NO_MEMLEAK_IF: changing this line to self.forward(x0)
        self.xsol = xsol  # NO_MEMLEAK_IF: removing this line or using xsol.detach()

    def function4functional(self, x):
        return self.a * x

class DummyModuleEngine(torch.nn.Module):
    def __init__(self, a):
        super().__init__()
        self.a = torch.nn.Parameter(a)

    def forward(self, x):
        return self.a * x

class DummyModule(torch.nn.Module):
    def __init__(self, a):
        super().__init__()
        self.engine = DummyModuleEngine(a)
        x0 = torch.ones_like(a)
        xsol = functional(self.engine.forward, x0)
        self.xsol = xsol
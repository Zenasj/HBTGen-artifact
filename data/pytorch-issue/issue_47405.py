class A():
    def __init__(self):
        pass

    def f(self, y: torch.device):
        return self.x.to(device=y)

    # def f(self, x, y: "Device"):  # works
        # return x.to(device=y)

def g():
    a = A()
    return a.f(torch.rand(3), torch.device("cpu"))

def f2(x, y: torch.device):
    return x.to(device=y)

script_f2 = torch.jit.script(f2)  # works
script_g = torch.jit.script(g)  #fails

import torch

@torch.jit.script
class A():
    def __init__(self):
        self.x = torch.rand(3)
        pass

    def f(self, y: torch.device):
        return self.x.to(device=y)

def g():
    a = A()
    return a.f(torch.device("cpu"))

script_g = torch.jit.script(g)
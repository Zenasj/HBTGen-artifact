import torch.nn as nn

import torch
from torch import nn

import torch._dynamo

import logging
torch._dynamo.config.log_level = logging.DEBUG
torch._dynamo.config.verbose = True

def check(module, attr: str) -> None:
    inp = torch.ones(1)
    compiled_value = m(inp) 
    eager_value = m._orig_mod(inp)
    prefix = "✅" if (compiled_value == eager_value).all().item() else "❌"
    print(f"{prefix} {attr}={getattr(m._orig_mod, attr)}: compiled={compiled_value}, eager: {eager_value}")

print("=== Foo attribute test ===")
class MyModuleFoo(nn.Module):

    foo: bool

    def __init__(self):
        super().__init__()
        self.foo = True

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        if self.foo:
            return x * 123
        else:
            return x * 0


m = torch.compile(MyModuleFoo())
check(m, "foo")
m._orig_mod.foo = False
check(m, "foo")

print("=== Training attribute test ===")
class MyModuleTraining(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        if self.training:
            return x * 123
        else:
            return x * 0

m = torch.compile(MyModuleTraining())
check(m, "training")
m._orig_mod.training = False
check(m, "training")
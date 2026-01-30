import torch
import torch.nn as nn

class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()
        self.a = 1
        self.b = 2

    def forward(self):
        return self.a + self.b

class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.sub = SubModule()

    def forward(self):
        return self.sub()

mod = torch.jit.script(Module())
mod.eval()
frozen_mod = torch.jit.freeze(mod, preserved_attrs = ['sub.a'])

mod.sub   # OK
mod.sub.a # OK
mod.sub.b # Error, not preserved
mod()     # = 3
mod.sub.a = 0
mod()     # = 2
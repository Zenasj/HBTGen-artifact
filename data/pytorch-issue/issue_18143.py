@jit.script_method
def forward(self, input):
    # type: (Tensor) -> Tensor
    for _ in range(2):
        for module in self.module_list:
            input = module(input)
    return input

import torch
import torch.nn as nn
import torch.jit as jit
from torch.jit import Tensor

class TestModule(jit.ScriptModule):
    __constants__ = ['module_list']

    def __init__(self, n):
        super().__init__()
        self.module_list = nn.ModuleList([DummyModule() for _ in range(n)])

    @jit.script_method
    def forward(self, input):
        # type: (Tensor) -> Tensor
        for _ in range(2):
            for module in self.module_list:
                input = module(input)
        return input

class DummyModule(jit.ScriptModule):
    @jit.script_method
    def forward(self, input):
        return input

if __name__ == "__main__":
    tensor = torch.tensor(1)
    module = TestModule(2)
    print(module(tensor))

def __init__(self, mod_list):
    super(M, self).__init__(False)
    self.mods = nn.Sequential(mod_list, mod_list)

@torch.jit.script_method
def forward(self, v):
    for m in self.mods:
        v = m(v)
    return v
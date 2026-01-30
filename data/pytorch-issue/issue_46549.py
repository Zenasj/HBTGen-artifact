import torch.nn as nn

import torch

class MyModule(torch.nn.Module):
    # module: torch.nn.ModuleDict
    def __init__(self):
        super().__init__()
        self.module = torch.nn.ModuleDict()
        self.module["AA"] = torch.nn.Linear(2, 2)

    @torch.jit.ignore
    def not_scripted(self, input):
        for name, module in self.module.items():
            return module(input)

    def forward(self, input):
        out = self.not_scripted(input)
        return out




input = torch.rand([2, 2])
print(input)
# scripted = torch.jit.script(MyModule())
scripted = torch.jit.script(MyModule())
scripted(input)
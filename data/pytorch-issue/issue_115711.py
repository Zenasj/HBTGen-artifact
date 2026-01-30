import torch.nn as nn

import torch

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 4)
        self.step = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 1
        self.step += 1
        return self.layer(x)

m = MyModule()
opt_m = torch.compile(backend="eager")(m)

x = torch.randn(3, 4)
print(opt_m(x))

x = torch.randn(4, 4)
print(opt_m(x))

x = torch.randn(5, 4)
print(opt_m(x))

self.step

nn.Module

nn.Module

torch.SymInt/Float

nn.Module

torch.mark_dynamic
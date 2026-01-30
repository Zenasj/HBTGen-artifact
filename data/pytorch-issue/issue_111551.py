import torch.nn as nn

import torch
from torch_geometric.nn.module_dict import ModuleDict

class SomeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module_dict = ModuleDict({
            ("author", "writes", "paper"): torch.nn.Linear(1, 1),
        })

    def forward(self, x):
        x = self.module_dict[("author", "writes", "paper")](x)
        return x

model = torch.compile(SomeModel())
model(torch.randn(100, 1))
import torch
import torch.nn as nn

from typing import Dict

class A(nn.Module):
    def __init__(self):
        super().__init__()
        self.x = tuple([0, 1, 2])
        self.y = nn.ModuleList([nn.Linear(10, 10)] * 3)

    def forward(self, x):
        ret = (x + 1, x + 2, x + 3)
        # return dict(zip(self.x, ret))  # fail
        return dict(list(zip(self.x, ret)))  # fail
        # return dict([(name, res) for name, res in zip(self.x, ret)])  # work

a = A()
script = torch.jit.script(a)
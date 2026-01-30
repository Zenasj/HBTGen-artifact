import torch.nn as nn

from torch.fx import subgraph_rewriter
from torch.fx import symbolic_trace
import torch

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attr = torch.tensor(3)

    def forward(self, x, y, z):
        return x + y + self.attr

class Pattern(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attr = torch.tensor([])

    def forward(self, x, y, z):
        return x + y + self.attr

class Replacement(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attr = torch.tensor([])

    def forward(self, x, y, z):
        return x + y - self.attr  

m = M()
m = symbolic_trace(m)
subgraph_rewriter.replace_pattern(m, Pattern(), Replacement())
print(m)
print(m.attr)
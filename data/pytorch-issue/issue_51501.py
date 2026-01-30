import torch.nn as nn

import torch
from torch.fx import symbolic_trace

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        return x + y[:, 1:10, ...]

x = torch.rand(1, 2, 3)
m = M()
m = symbolic_trace(m)
print(m(x, x))
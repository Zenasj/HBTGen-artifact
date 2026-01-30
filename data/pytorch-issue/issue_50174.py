import torch.nn as nn

import torch
from torch.fx import symbolic_trace
    
    
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.cat(x, dim=1)
        return x
    
m = M().eval()
m = symbolic_trace(m)
print(m)
print(m.graph)
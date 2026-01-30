import torch.nn as nn

import torch
from torch.fx import symbolic_trace

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y = 3):
        return x + y
    
m = symbolic_trace(M())
m(3)
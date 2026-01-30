import torch.nn as nn

import torch
from torch.fx import symbolic_trace

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        if y is None:
            # this path is not captured since when we trace y is a Proxy and it is not None
            y = x
        return x + y
    
m = M()
m = symbolic_trace(m)

x = torch.rand(1)
# this works
m(x, x)
# this does not work
m(x, None)
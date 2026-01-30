import torch.nn as nn

import torch
from torch.fx import symbolic_trace
class myMod(torch.nn.Module):
    def __init__(self):
        super(myMod, self).__init__()

    def forward(self, x, mask):
        return x.masked_fill(mask, float("-inf"))

m = myMod()
inp = torch.randn(4)
print(inp)
mask = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
print(m(inp, mask))
ms = symbolic_trace(m)
print(ms)
print(ms(inp, mask))
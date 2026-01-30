import torch.nn as nn

import torch
from torch import nn

class BadFirst(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_firsts = x[:, 0]
        print(f"x_firsts: {x_firsts}") # removing this line fixes the issue.
        return x_firsts


m = BadFirst().eval()
x = torch.rand(10, 5)

res = m(x) # this works
torch.onnx.export(m, x, "m.onnx") # this fails due to a resolve_conj op
import torch.nn as nn

import torch

class DimModel(torch.nn.Module):
    def forward(self, input):
        out = input * 2
        out *= out.dim()
        return out

m = torch.jit.script(DimModel())

input_1 = torch.arange(6).view(2, 3)
print(m(input_1))
"""
outputs tensor([[ 0,  4,  8],
        [12, 16, 20]])
"""

input_2 = torch.arange(6).view(1, 2, 3)
print(m(input_2))
"""
outputs tensor([[[ 0,  4,  8],
         [12, 16, 20]]])

but should be tensor([[[ 0,  6, 12],
         [18, 24, 30]]])
The correct result can also be produced if DimModel()(input_1) is commented
"""
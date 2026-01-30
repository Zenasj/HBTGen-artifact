import torch.nn as nn

import torch

class Mod(torch.nn.Module):
    def forward(self, x):
        y = torch.ops.aten.to(x, dtype=torch.float16, copy=True)
        y.mul_(2)
        return y

x = torch.randn(4)
m = torch.export.export(Mod(), (x,))
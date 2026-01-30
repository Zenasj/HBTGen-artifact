import torch.nn as nn

import torch

class MatMulModel(torch.nn.Module):
    def forward(self, x):
        return torch.mm(x, x) + x

x = torch.ones(3, 3)
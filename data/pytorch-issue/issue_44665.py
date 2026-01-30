import torch
from torch.fx import symbolic_trace
import torch.nn as nn

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def transpose_inp(self, x):
        new_shape = x.size()[:-1] + (5, 5)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        tmp = self.transpose_inp(x)
        return tmp

m = symbolic_trace(M())
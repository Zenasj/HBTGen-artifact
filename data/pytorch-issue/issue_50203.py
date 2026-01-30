import torch
import torch.nn as nn

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.addmm(x, y)

traced_module = symbolic_trace(M())
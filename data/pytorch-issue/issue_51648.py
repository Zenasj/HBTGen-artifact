import torch
import torch.nn as nn

class M(nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.W = torch.nn.Parameter(torch.randn(5))

    def forward(self, x):
        return torch.dot(self.W, x)

traced = fx.symbolic_trace(M())
traced(5)
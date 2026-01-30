import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx

class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.W = torch.nn.Parameter(torch.randn(5))
        self.ones = torch.tensor([1.])

    def forward(self, x):
        return torch.dot(self.W, x)


mod = fx.symbolic_trace(Test())(5)
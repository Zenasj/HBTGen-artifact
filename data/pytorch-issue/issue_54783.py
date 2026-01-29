# torch.rand(10, 10, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(10, 10, requires_grad=True))
        self.b1 = nn.Parameter(torch.rand(10, 10, requires_grad=True))
        self.b2 = nn.Parameter(torch.rand(10, 10, requires_grad=True))

    def forward(self, x):
        bias = self.b1 * self.b2
        return F.linear(x, self.weight, bias)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(10, 10, requires_grad=True)


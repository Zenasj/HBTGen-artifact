# torch.rand(2, 4, requires_grad=True) * 4 as a tuple
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        a, b, c, d = inputs
        x = a + b
        y = c + d
        y.copy_(x)  # Preserve in-place copy for FX graph reproduction
        x = torch.relu(x)
        x = x.cos()
        x = x.cos()
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return tuple(torch.randn(2, 4, requires_grad=True) for _ in range(4))


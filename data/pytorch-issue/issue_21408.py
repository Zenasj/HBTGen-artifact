# torch.rand(3, 3, dtype=torch.float32), torch.rand(3, 3, dtype=torch.float64)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        d = y.sum()  # Sum of Double tensor produces a Double scalar
        return x * d  # Float * Double → results in Float tensor (problem source)

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.rand(3, 3, dtype=torch.float32, requires_grad=True)
    y = torch.rand(3, 3, dtype=torch.float64, requires_grad=True)
    return (x, y)


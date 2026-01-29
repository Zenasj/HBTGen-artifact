# torch.rand(4), torch.tensor([3])  # x: (4,), y: (1,)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        z = y.item()
        torch._check(z == 3)
        return x + z

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.rand(4)
    y = torch.tensor([3])
    return (x, y)


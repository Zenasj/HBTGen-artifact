# torch.rand(4,4), torch.rand(4,4)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        return torch.cat([x, y], dim=0)

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.rand(4, 4, requires_grad=True), torch.rand(4, 4, requires_grad=True))


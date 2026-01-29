# Returns a tuple of (torch.rand(1, 1, requires_grad=True), torch.rand(2, 1, requires_grad=True)), dtype=torch.float32
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x1, x2 = inputs
        return torch.cdist(x1, x2)

def my_model_function():
    return MyModel()

def GetInput():
    x1 = torch.rand(1, 1, requires_grad=True)
    x2 = torch.rand(2, 1, requires_grad=True)
    return (x1, x2)


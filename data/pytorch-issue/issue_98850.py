# torch.rand(2, 3, 4), torch.rand(2, 3, 4)  # Two tensors with same shape
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, inputs):
        x, y = inputs
        z = torch.cat([x, y])  # Default dim=0, which triggers the bug
        z = torch.relu(z)
        return z

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.randn(2, 3, 4)
    y = torch.randn(2, 3, 4)
    return (x, y)


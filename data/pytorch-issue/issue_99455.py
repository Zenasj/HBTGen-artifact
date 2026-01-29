# torch.rand(1, dtype=torch.float32)  # Dummy input tensor
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, input):
        x = torch.randn(5, 5, requires_grad=True)
        y = x + 2
        return x, y

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.float32)


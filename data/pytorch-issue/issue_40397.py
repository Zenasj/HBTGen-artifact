# torch.rand(3, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch._softmax(x, -1, False)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(3, 3)


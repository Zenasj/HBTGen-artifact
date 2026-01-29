import torch
from torch import nn

# torch.rand(1, 3, 224, 224, dtype=torch.float32)
class MyModel(nn.Module):
    def forward(self, x):
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)


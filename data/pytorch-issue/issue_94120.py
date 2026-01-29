# torch.rand(0, 2**20, 8796093022207, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.clone(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand([0, 2**20, 8796093022207], dtype=torch.float32)


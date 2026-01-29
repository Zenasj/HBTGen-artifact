# torch.randint(0, 10, (1,), dtype=torch.long)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x * 2

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 10, (1,), dtype=torch.long)


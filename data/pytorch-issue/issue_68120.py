# torch.randint(0, 2, (1, 64), dtype=torch.long)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        out = (x == 0).all(1)
        return out

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 2, (1, 64), dtype=torch.long)


# torch.rand(2, 3, dtype=torch.float32, device="cuda")
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.cat(torch.split(x, 1), dim=-1)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, dtype=torch.float32, device="cuda")


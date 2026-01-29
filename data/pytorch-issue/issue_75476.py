# torch.rand(2, 2, dtype=torch.half, device="cuda")
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, p):
        p = torch.sigmoid(p)
        return p ** 2.0  # Matches gamma=2.0 from original function

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 2, dtype=torch.half, device="cuda")


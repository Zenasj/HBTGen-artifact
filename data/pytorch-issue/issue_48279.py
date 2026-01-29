# torch.rand(64, 64, 256, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        x = torch.exp(x)
        x = torch.exp(x)
        x = torch.exp(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(64, 64, 256, device='cuda', dtype=torch.float32)


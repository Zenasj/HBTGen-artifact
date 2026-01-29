# torch.rand(1, 2048, 2048, 4, dtype=torch.double)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.mean(x[..., :3])

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 2048, 2048, 4, dtype=torch.double, device='cuda')


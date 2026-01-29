# torch.rand((), dtype=torch.float32)  # 0D tensor input
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, a):
        result = torch.deg2rad(a).sin()
        out = torch.empty((128, 128), device=a.device).fill_(result)
        return out

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand((), dtype=torch.float32)


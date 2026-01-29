# torch.rand(1, 2, 8, 8, dtype=torch.float, device='cuda', memory_format=torch.channels_last)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.nn.functional.interpolate(x, size=16, mode='nearest')

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.randn(1, 2, 8, 8, dtype=torch.float, device='cuda') \
        .to(memory_format=torch.channels_last)
    x.requires_grad_()
    return x


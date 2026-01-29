# torch.rand(128, 256, dtype=torch.bfloat16, device='cpu', requires_grad=True)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.amax(x, dim=[-1], keepdim=False)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(128, 256, dtype=torch.bfloat16, device='cpu', requires_grad=True)


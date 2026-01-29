# (torch.randn(10000, 10000, device='cuda'), torch.randn(10000, 10000, device='cuda'))
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        a, b = inputs
        return a @ b

def my_model_function():
    return MyModel()

def GetInput():
    return (
        torch.randn(10000, 10000, device='cuda'),
        torch.randn(10000, 10000, device='cuda')
    )


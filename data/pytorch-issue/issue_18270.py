# torch.rand(5,5, dtype=torch.float, device='cuda'), torch.rand(5,5, dtype=torch.float, device='cuda')

import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x, y=None, c: float = 0.1):
        if y is not None:
            y.copy_(y * c + (1 - c) * x.detach())
        return x

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.randn(5, 5, device='cuda', requires_grad=True)
    y = torch.randn(5, 5, device='cuda', requires_grad=False)
    return (x, y)


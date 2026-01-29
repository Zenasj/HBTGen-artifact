# torch.randn(3,4, device='cuda'), torch.randn(4, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        a, b = inputs
        return b.expand([1, a.shape[0], b.shape[-1]])

def my_model_function():
    return MyModel()

def GetInput():
    a = torch.randn(3, 4, device='cuda')
    b = torch.randn(4, device='cuda')
    return (a, b)


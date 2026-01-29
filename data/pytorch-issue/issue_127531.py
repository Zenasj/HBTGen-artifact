# torch.rand(5, dtype=torch.float32, device='cuda'), torch.rand(5, dtype=torch.float32, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        a, b = inputs
        return a + b

def my_model_function():
    return MyModel()

def GetInput():
    return (
        torch.randn(5, dtype=torch.float32, device='cuda'),
        torch.randn(5, dtype=torch.float32, device='cuda')
    )


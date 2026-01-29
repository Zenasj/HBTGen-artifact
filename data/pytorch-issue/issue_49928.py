# (torch.rand(4, 512, 32, dtype=torch.float32), torch.rand(4, 512, 32, dtype=torch.float32))
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        a, b = inputs
        c = torch.cdist(a, b, p=1.0)  # Reproduces the reported CUDA error when input size reaches 512
        return c.mean()

def my_model_function():
    return MyModel()

def GetInput():
    a = torch.randn(4, 512, 32, requires_grad=True).cuda()  # Matches problematic input size from original issue
    b = torch.randn(4, 512, 32, requires_grad=True).cuda()
    return (a, b)


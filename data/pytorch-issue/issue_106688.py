# Inputs: 6 tensors (3x (5,4) and 3x (4,3)), all on CUDA
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x1, x2, x3, w1, w2, w3 = inputs
        x = torch.stack([x1, x2, x3])
        w = torch.stack([w1, w2, w3])
        return torch.bmm(x, w)

def my_model_function():
    return MyModel()

def GetInput():
    x1 = torch.randn(5, 4, device='cuda')
    x2 = x1 + 1
    x3 = x1 + 2
    w1 = torch.randn(4, 3, device='cuda')
    w2 = w1 + 1
    w3 = w1 + 2
    return (x1, x2, x3, w1, w2, w3)


# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        v1 = torch.matmul(x, x.transpose(-1, -2))
        v2 = v1 / -0.0001
        v3 = v2.softmax(dim=-1)
        v4 = torch.matmul(v3, x)
        return v4

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 3, 64, 64)


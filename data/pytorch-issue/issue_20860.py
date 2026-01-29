# torch.rand(B, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.bmm(x, x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(32, 60, 60, dtype=torch.float32).cuda()


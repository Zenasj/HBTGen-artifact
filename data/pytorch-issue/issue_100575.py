# torch.rand(B, C, H, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x.repeat_interleave(2, dim=-1)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand((2, 2, 16))


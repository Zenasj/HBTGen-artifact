# torch.rand(65536, dtype=torch.bfloat16)  # each tensor in the input tuple has this shape and dtype
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        return (x + y).relu()

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.randn(65536).cuda().bfloat16()
    y = torch.randn_like(x)
    return (x, y)


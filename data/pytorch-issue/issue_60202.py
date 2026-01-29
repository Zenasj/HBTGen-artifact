# Input: tuple of two tensors each with shape (1, 256, 256, 256) and dtype=torch.float32
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        return x + y

def my_model_function():
    return MyModel()

def GetInput():
    return (
        torch.rand(1, 256, 256, 256, dtype=torch.float32),
        torch.rand(1, 256, 256, 256, dtype=torch.float32)
    )


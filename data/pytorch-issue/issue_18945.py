# torch.rand(10, 4, dtype=torch.float32), torch.rand(10,4, dtype=torch.float32)  # input (x, y)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        # Custom implementation of binary_cross_entropy with double-backward support
        loss = -(x.log() * y + (1 - x).log() * (1 - y)).mean()
        return loss

def my_model_function():
    return MyModel()

def GetInput():
    B, C = 10, 4
    x = torch.rand(B, C, dtype=torch.float32, requires_grad=True)
    y = torch.rand(B, C, dtype=torch.float32)
    return (x, y)


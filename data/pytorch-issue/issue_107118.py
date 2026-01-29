# torch.rand(1, 4, 32, 32, dtype=torch.float32)  # First input; second input is torch.empty(0, ...)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        return torch.cat([x, y], dim=1)

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.rand(1, 4, 32, 32, dtype=torch.float32, device='cuda')
    y = torch.empty(0, dtype=torch.float32, device='cuda')
    return (x, y)


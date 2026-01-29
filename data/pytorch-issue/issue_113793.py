# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        with torch.no_grad():
            view_x = x[:]
            view_x.add_(1.0)
        return view_x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 1, 1, dtype=torch.float32, requires_grad=True)


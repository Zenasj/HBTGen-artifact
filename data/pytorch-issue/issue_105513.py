# torch.rand(4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        with torch.no_grad():
            assert x.max() < 5, f"invalid max {x.max()}"
            x = torch.sin(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4)


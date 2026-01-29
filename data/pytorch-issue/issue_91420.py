# torch.rand(1, 1, 1, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        sorted_vals, _ = torch.sort(x, stable=True, dim=1, descending=True)
        return sorted_vals

def my_model_function():
    return MyModel()

def GetInput():
    v0 = torch.scalar_tensor(True, dtype=torch.float)
    v3 = v0.expand([1, 1, 1])
    return v3


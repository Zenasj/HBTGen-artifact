# torch.rand(B, 1, dtype=torch.bool)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.ops.aten._is_all_true(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 2, (2, 1), dtype=torch.bool)


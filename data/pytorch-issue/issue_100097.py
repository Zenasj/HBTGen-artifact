# torch.randint(42, (17,), dtype=torch.int64)  # Inferred input shape and dtype
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, a):
        return torch.sum(a, dtype=torch.int32)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(42, (17,), dtype=torch.int64)


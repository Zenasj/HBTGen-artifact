# torch.randint(1, 11, (), dtype=torch.int64)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        a = x.item()
        torch._constrain_as_size(a, min=1, max=10)
        return torch.ones(a, a, dtype=x.dtype)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(1, 11, (), dtype=torch.int64)


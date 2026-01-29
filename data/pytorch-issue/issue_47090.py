# torch.rand(2, dtype=torch.float32)
import torch
from torch import nn

class _T(torch.Tensor):
    pass

class MyModel(nn.Module):
    def forward(self, x):
        subclass_x = x.as_subclass(_T)
        return torch.max(subclass_x, dim=0)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, dtype=torch.float32)


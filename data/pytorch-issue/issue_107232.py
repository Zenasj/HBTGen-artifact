# torch.rand(3, 3, dtype=torch.float32)
import torch
from torch import nn
from typing import Union

class MyModel(nn.Module):
    def forward(self, x):
        def f(a) -> Union[torch.Tensor, int]:  # Problematic type annotation
            if a.ndim == 2:
                return a
            return 3
        return f(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 3, dtype=torch.float32)


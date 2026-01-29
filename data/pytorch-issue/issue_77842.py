# torch.rand(1, 2, 3, dtype=torch.int32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return (
            x.cumsum(0, dtype=None),
            x.cumsum(0, dtype=torch.int32),
            x.cumsum(0, dtype=torch.float)
        )

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 2, 3).to(torch.int32)


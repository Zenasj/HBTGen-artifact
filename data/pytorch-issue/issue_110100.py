# torch.rand(5, 2, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        out = self.linear(x)
        return out

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(5, 2, 2)


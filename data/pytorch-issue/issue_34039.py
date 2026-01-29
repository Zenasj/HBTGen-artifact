# torch.rand(1, 64, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 32)  # Matches FBGEMM's example NC-64 parameter

    def forward(self, x):
        return self.linear(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 64, dtype=torch.float32)


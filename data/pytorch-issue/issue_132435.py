# torch.rand(B, 16, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.rand((1, 16)))

    def forward(self, x):
        return self.param * x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 16)  # Batch size 2, matching minimal repro's context


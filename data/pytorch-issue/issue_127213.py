# torch.rand(10, 10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(10, 10).cuda()  # Matches the CUDA setup in the original test


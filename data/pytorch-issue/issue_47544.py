# torch.rand(B, 2, dtype=torch.float)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.randn(2))
        self.linear = nn.Linear(2, 2)
        self.register_buffer('attr', torch.randn(2))
        self.register_buffer('attr2', torch.randn(2))

    def forward(self, x):
        return self.linear(self.W + (self.attr + self.attr2) + x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(5, 2)  # Example input with batch size 5


# torch.rand(16, dtype=torch.float32)  # Input shape is (16,)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.F = nn.Parameter(torch.rand(16, 16))  # 2D parameter causing buffer size issues on MPS
        
    def forward(self, x):
        return x @ self.F  # Matrix multiply (1D input @ 2D parameter)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(16)  # 1D tensor of shape (16,)


# torch.rand(B, 2, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.my_param = torch.nn.Parameter(torch.randn([2, 3]))  # Matches parameter shape from original issue

    def forward(self, x):
        return x + self.my_param  # Addition with parameter as in original issue

def my_model_function():
    return MyModel()  # Returns initialized model instance

def GetInput():
    # Returns batched input (B=2) with shape (2, 2, 3) to allow dynamic batch dimension
    return torch.rand(2, 2, 3, dtype=torch.float32)  # dtype matches parameter's float32


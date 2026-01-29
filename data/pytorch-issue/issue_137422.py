# torch.rand(3, 1, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Correct usage of squeeze without `dim=None` to avoid runtime error
        return torch.squeeze(x)  # Removes singleton dimensions without specifying dim

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(3, 1, 2, dtype=torch.float32)


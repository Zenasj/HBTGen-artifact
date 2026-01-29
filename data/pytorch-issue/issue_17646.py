# torch.rand(2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        # Division operation that may cause kernel crash in older PyTorch versions (e.g., 1.0.0)
        return x / 0.1  # Original problematic operation described in the issue

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a 1D tensor of shape (2,) matching the input expected by MyModel
    return torch.rand(2, dtype=torch.float32)


# torch.rand(1, dtype=torch.int64)  # Input is a 1-element tensor containing the integer end value for arange
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        y = x.item()
        torch._check_is_size(y)  # Preserving original check from the issue
        r = torch.arange(y, dtype=torch.float32)
        torch._check(r.size(0) == y)  # Symbolic shape inference failure point
        return r

def my_model_function():
    return MyModel()

def GetInput():
    return torch.tensor([300])  # Matches the original test input in the issue


# torch.rand(3, dtype=torch.float32)  # Input shape: (start, end, step) as 0D tensors
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, params):
        start = params[0]   # 0D tensor (scalar)
        end = params[1]     # 0D tensor (scalar)
        step = params[2]    # 0D tensor (scalar)
        return torch.arange(start=start, end=end, step=step)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random start, end, step as 0D tensors (shape (3,))
    return torch.rand(3, dtype=torch.float32)


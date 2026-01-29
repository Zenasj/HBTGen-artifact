# torch.rand(1, 199, dtype=torch.float32)  # Matches problematic tensor shape from issue
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Workaround for MPS buffer issue: use .zero_() instead of in-place assignment
        return x.zero_()  # Returns tensor of zeros (avoids buffer error)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a tensor that would trigger buffer error with in-place assignment (x[:] = 0)
    return torch.rand(1, 199, dtype=torch.float32)


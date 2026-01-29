# torch.rand(2, dtype=torch.float32)
import torch
import numpy as np
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        a, b = x[0], x[1]
        min_ab = torch.min(a, b)
        min_ba = torch.min(b, a)
        # Return True if results differ (NaN comparison uses inequality)
        return torch.ne(min_ab, min_ba)

def my_model_function():
    return MyModel()

def GetInput():
    # Create tensor with 0 and NaN to trigger order-dependent behavior
    return torch.tensor([0., float('nan')], dtype=torch.float32)


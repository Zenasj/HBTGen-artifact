# torch.rand(1, dtype=torch.float32)
import torch
import math
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Problematic usage of math.exp on tensor (breaks gradients)
        scalar_value = math.exp(x.item())
        return torch.tensor([scalar_value], dtype=x.dtype)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a scalar tensor with gradients enabled
    return torch.rand(1, requires_grad=True, dtype=torch.float32)


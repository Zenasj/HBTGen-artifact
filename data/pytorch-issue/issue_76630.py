# torch.rand(5, dtype=torch.float32), torch.rand(5,5, dtype=torch.float64)  # inferred input shapes and dtypes
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        input, min_tensor = inputs
        r1 = torch.maximum(input, min_tensor)  # Promotes dtype like max/min
        r2 = torch.clamp(input, min=min_tensor)  # Does NOT promote dtype like clamp
        return r1, r2  # Return both outputs to compare dtypes

def my_model_function():
    return MyModel()

def GetInput():
    input = torch.rand(5, dtype=torch.float32)
    min_tensor = torch.rand(5, 5, dtype=torch.float64)
    return (input, min_tensor)  # Returns tuple of input and min_tensor


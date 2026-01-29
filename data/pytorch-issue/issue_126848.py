# torch.rand((), dtype=torch.float32)  # Inferred input shape is a 0D tensor
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x * 2.0 ** (-127 + 3)  # Core operation causing the denormal issue

def my_model_function():
    return MyModel()  # Returns the model instance with the problematic operation

def GetInput():
    # Returns a scalar tensor (0D) that triggers denormal behavior
    return torch.tensor(0.1875, dtype=torch.float32)


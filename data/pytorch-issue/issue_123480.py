# torch.rand(2, 3, 4, dtype=torch.float32)  # Inferred input shape based on sum() usage pattern
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Reproduces the export bug with empty dim list in sum operation
        return torch.sum(x, dim=[])  # Uses dim=[] to trigger the serialization issue

def my_model_function():
    # Returns the model instance with default initialization
    return MyModel()

def GetInput():
    # Generates random input tensor matching the expected shape and dtype
    return torch.rand(2, 3, 4, dtype=torch.float32)


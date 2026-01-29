# torch.rand(1000, 1, 1, 1, dtype=torch.float32)  # Matches the example's input shape as 4D tensor
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        chunks = x.split(10, dim=0)  # Split along first dimension (default for 4D tensor)
        for chunk in chunks:
            chunk.add_(1)  # In-place addition to trigger overlap checks
        return chunks[0]  # Return first chunk as in original function

def my_model_function():
    return MyModel()  # Returns the model instance

def GetInput():
    return torch.rand(1000, 1, 1, 1, dtype=torch.float32)  # Reproduces the original input's shape in 4D format


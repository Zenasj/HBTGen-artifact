# torch.rand(2, dtype=torch.float32)  # Input shape is 1D tensor of size 2
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Use native torch.split to avoid custom Function issues
        return torch.split(x, split_size_or_sections=1, dim=0)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a 1D tensor of size 2 as required by the model
    return torch.rand(2, dtype=torch.float32)


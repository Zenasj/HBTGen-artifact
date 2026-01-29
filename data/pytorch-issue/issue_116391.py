# torch.rand(128, 256, 6, 24, dtype=torch.float32)  # Inferred input shape from error logs
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Perform unbind along dimension 2 (as per error logs showing shape (128,256,6,24))
        unbinded = torch.unbind(x, dim=2)
        # Access first element of the tuple via getitem (causing backend compatibility issue)
        return unbinded[0]

def my_model_function():
    return MyModel()

def GetInput():
    # Generate input matching the problematic shape from the error logs
    return torch.rand(128, 256, 6, 24, dtype=torch.float32)


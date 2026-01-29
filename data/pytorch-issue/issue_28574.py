# torch.rand(B, C, H, W, dtype=torch.float32)  # Example: (1, 1, 1, 1)

import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, input):
        # Reshape input to match 2D tensor shape from the original issue
        x = input.view(1, 1)
        y = torch.zeros_like(x)
        # Replicate the problematic assignment causing segmentation fault
        y[:, 0] = x[:, 0]
        return y

def my_model_function():
    return MyModel()

def GetInput():
    # Return a 4D tensor matching expected input shape (B, C, H, W)
    return torch.rand(1, 1, 1, 1, dtype=torch.float32)


# torch.rand(8, 6, 8, 6, 6, 1, 3, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Apply digamma function
        x = torch.special.digamma(x)
        # Compute argmax over all elements (default behavior)
        x = torch.argmax(x)
        return x

def my_model_function():
    # Return an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Generate random input matching the required shape
    return torch.rand([8, 6, 8, 6, 6, 1, 3], dtype=torch.float32)


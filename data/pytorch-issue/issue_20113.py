# torch.rand(B, 10, 5, dtype=torch.float32)  # Inferred input shape based on the example in the issue

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Explicitly set p=2 to avoid the JIT issue
        return torch.norm(x, p=2, dim=1, keepdim=True)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B = 1  # Batch size
    C = 10  # Number of channels
    H = 5  # Height
    return torch.rand(B, C, H, dtype=torch.float32)


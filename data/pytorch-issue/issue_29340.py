# torch.rand(2, 2, dtype=torch.float32)  # Inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Use the correct function to check for finite values
        is_finite = torch.isfinite(x)
        return is_finite

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(2, 2, dtype=torch.float32)


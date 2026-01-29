# torch.rand(1, 10, dtype=torch.float32)  # Inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Use float(2) to ensure the output type is float
        return torch.pow(float(2), x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(1, 10)


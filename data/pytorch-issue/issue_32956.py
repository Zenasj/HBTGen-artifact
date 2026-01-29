# torch.rand(1, 2, dtype=torch.float32)  # Inferred input shape from the issue

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Use torch.narrow instead of slicing to ensure compatibility with ONNX and ONNX.js
        x = torch.narrow(x, dim=1, start=1, length=1)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 2, dtype=torch.float32)


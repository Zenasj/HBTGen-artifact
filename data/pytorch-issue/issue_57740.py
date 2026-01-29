# torch.rand(1, dtype=torch.float32)  # Dummy input tensor of shape (1,)

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("test", torch.tensor([1.0], dtype=torch.float32))  # Initialize buffer in __init__

    def forward(self, x):
        # Modify buffer value (as per the suggested fix in comments)
        self.test = torch.tensor([1.0, 3.0], dtype=torch.float32)  # Example modification
        return self.test

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1)  # Dummy input tensor matching the required shape


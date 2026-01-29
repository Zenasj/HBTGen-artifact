# torch.randint(0, 2049, (2**20,), dtype=torch.int32)  # Example input with shape (N,)

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.bincount(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a large tensor of integers with uniform values to mimic the issue's test case
    return torch.ones(2**20, dtype=torch.int32) * 2048


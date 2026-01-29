# torch.rand(1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        a = [1, 2, 3, 4]  # Reproducer list from the issue
        reversed_a = a[::-1]  # Reverse slicing that triggers the error
        return torch.tensor(reversed_a, dtype=torch.float32)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1)  # Dummy input to satisfy model signature requirements


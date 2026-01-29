# torch.rand(1, 1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # Matches the original model's structure

    def forward(self, x):
        return self.linear(x)

def my_model_function():
    return MyModel()  # Returns the standard model instance

def GetInput():
    return torch.randn(1, 1, dtype=torch.float32)  # Valid input for uncompiled model


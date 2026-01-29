# torch.rand(B, 32, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Matches the parameter shape (16, 32) from the issue example
        self.linear = nn.Linear(32, 16)  # Weight shape (16, 32)
    
    def forward(self, x):
        return self.linear(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random input tensor of shape (batch_size, 32)
    return torch.rand(2, 32)  # batch_size=2 is arbitrary but valid


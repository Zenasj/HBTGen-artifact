# torch.rand(B, 10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)  # Matches the hook examples using nn.Linear

    def forward(self, x):
        return self.linear(x)

def my_model_function():
    # Returns model with initialized weights
    return MyModel()

def GetInput():
    # Returns random input tensor matching the model's expected input shape
    B = 2  # Batch size (example value)
    return torch.rand(B, 10, dtype=torch.float32)


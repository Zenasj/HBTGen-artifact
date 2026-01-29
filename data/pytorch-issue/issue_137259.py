# torch.rand(1, 2000, dtype=torch.float32)  # Inferred input shape from DIM=2000 in original code
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2000, 2000)  # Matches original Net class structure

    def forward(self, x):
        return self.fc1(x)

def my_model_function():
    # Returns model instance with default initialization
    return MyModel()

def GetInput():
    # Returns random input matching expected shape (B=1, C=2000)
    return torch.rand(1, 2000, dtype=torch.float32)


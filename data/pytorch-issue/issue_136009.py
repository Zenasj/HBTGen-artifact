# torch.rand(20, 10, dtype=torch.float32)  # Inferred input shape (batch_size=20, features=10)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 1)  # Matches the original model structure in the issue

    def forward(self, x):
        return self.linear(x)

def my_model_function():
    # Returns the model instance with default initialization
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the expected shape (B=20, C=10)
    return torch.rand(20, 10)  # dtype=torch.float32 by default


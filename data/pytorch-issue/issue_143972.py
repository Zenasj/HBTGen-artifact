# torch.rand(B, 32, dtype=torch.float32)  # Assuming input is 2D (batch, features)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simple linear layer to mimic model structure
        self.linear = nn.Linear(32, 16)
        
    def forward(self, x):
        return self.linear(x)

def my_model_function():
    # Returns a model instance with default initialization
    return MyModel()

def GetInput():
    # Returns a random tensor matching the model's input expectation
    return torch.rand(8, 32, dtype=torch.float32)  # Batch size 8, 32 features


# torch.rand(B, 10, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        self.example_attr = "test"  # Attribute to demonstrate __getattr__ access

    def forward(self, x):
        return self.fc(x)

def my_model_function():
    # Returns a model instance with __getattr__-compatible attribute structure
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the model's expected input shape
    return torch.rand(2, 10, dtype=torch.float32)


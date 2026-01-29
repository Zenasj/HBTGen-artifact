# torch.rand(B, 5, dtype=torch.float32)  # Inferred input shape based on Dataset length (5 elements)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple linear layer to process input of shape (B, 5)
        self.linear = nn.Linear(5, 1)
    
    def forward(self, x):
        return self.linear(x)

def my_model_function():
    # Returns a simple model that processes 5-element inputs
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the model's expected input shape
    return torch.rand(1, 5, dtype=torch.float32)


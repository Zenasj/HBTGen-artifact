# torch.rand(32, 1024, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1024, 2048)
    
    def forward(self, x):
        # Simulates matmul operations that would trigger Triton codegen
        return self.linear(x)

def my_model_function():
    # Returns a model with default initialization
    return MyModel()

def GetInput():
    # Generates a random input tensor matching the model's expected shape
    return torch.rand(32, 1024)


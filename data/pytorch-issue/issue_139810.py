# torch.rand(B, 10, dtype=torch.float32)  # Assumed input shape based on common linear layer use cases
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)  # Matches input size in GetInput()
    
    def forward(self, x):
        return self.linear(x)

def my_model_function():
    # Returns initialized model instance
    return MyModel()

def GetInput():
    B = 1  # Batch size inferred from benchmark context
    return torch.rand(B, 10, dtype=torch.float32)


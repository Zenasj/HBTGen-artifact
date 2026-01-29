# torch.rand(B, 3, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(3, 1)  # Matches the model from issue example
    
    def forward(self, x):
        return self.linear(x)

def my_model_function():
    return MyModel()  # Returns the model instance with default initialization

def GetInput():
    B = 4  # Example batch size, arbitrary positive integer
    return torch.rand(B, 3, dtype=torch.float32)  # Matches input shape for nn.Linear(3, 1)


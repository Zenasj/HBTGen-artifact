# torch.rand(B, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mod = nn.Linear(3, 3, bias=False)  # Matches Linear(3,3,bias=False) in the original example
        
    def forward(self, x):
        return self.mod(x)

def my_model_function():
    # Returns the model instance with default initialization
    return MyModel()

def GetInput():
    # Returns input tensor of shape (3, 3) matching Linear layer's expected input
    return torch.rand(3, 3)


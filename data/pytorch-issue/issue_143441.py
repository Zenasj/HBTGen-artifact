# torch.rand(B, 3, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(3, 4)  # Matches SimpleModel's structure from the example
    
    def forward(self, x):
        return self.net(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random tensor matching the input expected by MyModel's Linear layer (3 features)
    return torch.rand(2, 3, dtype=torch.float)


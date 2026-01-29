# torch.rand(B, 16, dtype=torch.float32).cuda(0)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 8)  # First layer (input size 16 → 8)
        self.fc2 = nn.Linear(8, 4)   # Second layer (8 → 4)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def my_model_function():
    # Returns the base model (not wrapped in Pipe yet)
    return MyModel()

def GetInput():
    # Matches input shape (B=16, C=16) from the issue's example
    return torch.rand(16, 16, dtype=torch.float32).cuda(0)


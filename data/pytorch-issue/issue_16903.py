# torch.rand(B, 1, 10, 10, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10 * 10, 10)  # Input shape: (B, 1, 10, 10) → flattened to (B, 100)

    def forward(self, x):
        # Flatten the input tensor for linear layer
        x = x.view(x.size(0), -1)
        return self.linear(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random tensor matching the expected input shape (B, 1, 10, 10)
    return torch.rand(1, 1, 10, 10, dtype=torch.float)


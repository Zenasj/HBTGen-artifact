# torch.rand(B, 2, 1, 1, dtype=torch.float32)  # Inferred input shape from the example
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(2, 2)  # Matches the original Linear layer in the issue's code

    def forward(self, x):
        # Flatten 4D input (B, C, H, W) to 2D (B, C*H*W) to match Linear layer's expectation
        x = x.view(x.size(0), -1)
        return self.l(x)

def my_model_function():
    return MyModel()  # Returns the model instance with default initialization

def GetInput():
    # Returns a random 4D tensor matching the inferred input shape
    return torch.randn(2, 2, 1, 1)  # B=2, C=2, H=1, W=1 (matches example's torch.randn(2, 2))


# torch.rand(B, C, H, W, dtype=torch.float32)  # Example shape: (2, 2, 4, 2) → product B*C*H*W must equal 32
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.y = nn.Parameter(torch.randn(32))  # Matches product of input dimensions (32)

    def forward(self, x):
        # Reshape input to 1D and add parameter (requires x's element count to match self.y's length)
        return x.view(-1) + self.y

def my_model_function():
    # Returns model instance with fixed parameter (32 elements)
    return MyModel()

def GetInput():
    # Generates 4D tensor with product 32 (e.g., 2x2x4x2 = 32)
    return torch.randn(2, 2, 4, 2, dtype=torch.float32)


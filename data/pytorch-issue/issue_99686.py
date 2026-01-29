# torch.rand(1, 8, 1, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Split into chunks of [3, 2, 3] along channel dimension (dim=1)
        parts = torch.split(x, [3, 2, 3], dim=1)
        # Reorder the chunks and concatenate
        return torch.cat([parts[1], parts[0], parts[2]], dim=1)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a 4D tensor matching the expected input shape (B, C, H, W)
    return torch.randn(1, 8, 1, 1, dtype=torch.float32)


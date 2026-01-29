# torch.rand(B, 1, 1, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Simulate a random operation that would be blocked by vmap's DispatchKey::VmapMode
        return x.normal_()  # In-place normal_() to trigger vmap's random function restriction

def my_model_function():
    return MyModel()

def GetInput():
    # Return a 4D tensor matching the input shape (B, C, H, W) with dummy dimensions
    return torch.rand(2, 1, 1, 1)  # B=2, C=1, H=1, W=1 (minimal shape based on example context)


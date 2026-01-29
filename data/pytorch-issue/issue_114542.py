# torch.rand(16, 1, dtype=torch.int64, device='cuda')

import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Register shifts tensor as buffer for reproducibility
        self.register_buffer('shifts', torch.arange(0, 32, 8, dtype=torch.int64, device='cuda'))
    
    def forward(self, x):
        # Perform bitwise shift and view operation
        shifted = x >> self.shifts
        # Explicit view to target shape (16,4)
        return shifted.view(16, 4)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate input matching the required shape and dtype
    return torch.randint(
        2,
        size=(16, 1),
        dtype=torch.int64,
        device="cuda"
    )


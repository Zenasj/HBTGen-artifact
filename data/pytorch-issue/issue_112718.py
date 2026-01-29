# torch.rand(2, 3, 5, 4, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = nn.Identity()  # Placeholder for model components before unique_consecutive

    def forward(self, x):
        processed = self.identity(x)
        # Apply unique_consecutive along dim=2 as in the issue's code
        processed = processed.unique_consecutive(dim=2)
        return processed

def my_model_function():
    return MyModel()

def GetInput():
    # Matches input shape (B, C, H, W) with dim=2 (H) for unique_consecutive
    return torch.rand(2, 3, 5, 4, dtype=torch.float32)


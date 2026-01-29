# torch.rand(B, 1, 4, 4, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift_size = 1  # Based on the example's shifts of (-1, -1)

    def forward(self, x):
        # Apply torch.roll on the last two dimensions (H, W) of 4D input
        return torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a 4D tensor matching the expected input shape (B, C, H, W)
    return torch.rand(1, 1, 4, 4)


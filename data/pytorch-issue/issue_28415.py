# torch.rand(B, C, H, W, dtype=torch.float32)  # e.g., (2, 3, 4, 4)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, channels):
        super(MyModel, self).__init__()
        self.channels = channels

    def forward(self, x):
        # Generate permutation tensor on the same device as input
        perm = torch.randperm(self.channels, device=x.device)
        return x[:, perm, :, :]

def my_model_function():
    # Initialize model with 3 input channels (common for RGB images)
    return MyModel(channels=3)

def GetInput():
    # Generate random input with shape (B, C, H, W)
    B, C, H, W = 2, 3, 4, 4
    return torch.rand(B, C, H, W, dtype=torch.float32)


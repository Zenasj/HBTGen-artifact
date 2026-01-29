# torch.rand(B, C, H, W, dtype=torch.float32, device='cuda')  # Input shape: (128, 3, 14, 14)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(196, 256)
        self.alpha = nn.Parameter(torch.randn(1, 1, 3, device='cuda'))  # Initialized as learnable parameter

    def forward(self, x):
        flatten_1 = torch.flatten(x, 2)  # Flatten from dim=2
        transpose_2 = torch.transpose(flatten_1, 1, 2)  # Swap dims 1 and 2
        add_3 = torch.add(self.alpha, transpose_2)  # Add with broadcasting
        transpose_4 = torch.transpose(add_3, 1, 2)  # Swap dims back
        return self.linear_layer(transpose_4)  # Apply linear layer to last dimension

def my_model_function():
    # Returns initialized model on CUDA
    return MyModel().cuda()

def GetInput():
    # Returns random input matching expected shape and device
    return torch.randn(128, 3, 14, 14, dtype=torch.float32, device='cuda', requires_grad=False)


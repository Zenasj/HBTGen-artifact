# torch.rand(B, 10, dtype=torch.float, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Use linspace on CUDA (fixed in newer PyTorch versions)
        self.register_buffer('linspace_tensor', torch.linspace(-1, 1, 10, device='cuda'))
    
    def forward(self, x):
        # Add the precomputed linspace tensor (shape [10]) to input x (shape [B, 10])
        return x + self.linspace_tensor  # Broadcasting over batch dimension

def my_model_function():
    return MyModel()

def GetInput():
    # Return a random tensor matching the input shape (B=5 is arbitrary here)
    return torch.rand(5, 10, dtype=torch.float, device='cuda')


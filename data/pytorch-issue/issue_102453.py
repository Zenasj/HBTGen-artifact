# torch.rand(B, C, D, H, W, dtype=torch.float32, device='mps')
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        # Apply padding with mode='replicate' causing the MPS error in older PyTorch versions
        return torch.nn.functional.pad(x, (5, 5, 5, 5, 5, 5), mode='replicate')

def my_model_function():
    return MyModel()

def GetInput():
    # Create input tensor matching the 5D shape (N,C,D,H,W) required by MyModel
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    return torch.rand(1, 1, 3, 32, 32, dtype=torch.float32, device=device)


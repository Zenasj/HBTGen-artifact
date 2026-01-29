# torch.rand(B, C, H, W, dtype=torch.float32, device='cuda')  # Inferred input shape (e.g., 1x1x3x3)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return torch.sin(x) + torch.cos(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 3, 3, dtype=torch.float32, device='cuda')


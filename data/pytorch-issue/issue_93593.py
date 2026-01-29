# torch.rand(1, 128, 1024, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, add):
        # Corrected to return flat tuple (var, mean) instead of wrapping in another tuple
        var, mean = torch.var_mean(add, dim=[2], correction=0, keepdim=True)
        return var, mean

def my_model_function():
    return MyModel()

def GetInput():
    # Matches the shape (1, 128, 1024) from the issue's args
    return torch.rand(1, 128, 1024, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')


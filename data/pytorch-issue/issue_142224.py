# torch.rand(64, 64, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return F.gelu(x)

def my_model_function():
    return MyModel()

def GetInput():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.randn(64, 64, device=device, dtype=torch.float32)


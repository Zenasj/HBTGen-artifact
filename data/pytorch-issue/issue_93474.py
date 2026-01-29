# torch.rand(4, 9, 10, dtype=torch.float32, requires_grad=True, device='cuda') 
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(5)
        
    def forward(self, x):
        return self.pool(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, 9, 10, dtype=torch.float32, requires_grad=True, device='cuda')


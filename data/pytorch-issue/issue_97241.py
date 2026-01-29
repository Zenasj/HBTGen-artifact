# torch.rand(3, 3, dtype=torch.float32, device='cuda', requires_grad=True)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(x).sum()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 3, dtype=torch.float32, device='cuda', requires_grad=True)


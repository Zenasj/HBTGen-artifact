# torch.rand(2048, dtype=torch.float32, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        isinf = torch.isinf(x)
        return [isinf]

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2048, dtype=torch.float32, device='cuda')


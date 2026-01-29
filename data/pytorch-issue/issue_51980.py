# torch.rand(5, 5, 5, dtype=torch.cdouble, device='cuda')  # Inferred input shape and dtype
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        cond = torch.zeros_like(x, dtype=torch.bool)
        return torch.where(cond, x, complex(0.0))

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(5, 5, 5, dtype=torch.cdouble, device='cuda')


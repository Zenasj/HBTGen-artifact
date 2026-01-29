# torch.rand(10, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Compare argmin outputs with keepdim=True vs keepdim=False
        val1 = torch.argmin(x, dim=None, keepdim=False)
        val2 = torch.argmin(x, dim=None, keepdim=True)
        # Check if values are equal after squeezing the keepdim=True result
        return torch.eq(val1, val2.squeeze()).all()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(10, 3)


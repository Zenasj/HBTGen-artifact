# torch.rand((), dtype=torch.int64, device='cuda')
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        minus_x = -x
        minus_x_float = minus_x.to(dtype=torch.float32)
        return torch.abs(minus_x_float)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(1, 10, size=(), dtype=torch.int64, device='cuda')


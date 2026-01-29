# torch.rand(4,4, dtype=torch.bfloat16)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        return torch.rrelu(x, 0.2, 0.8, training=True)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(4, 4, dtype=torch.bfloat16, requires_grad=True)


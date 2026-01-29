# torch.rand(1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return 2 * x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.tensor([1.0], requires_grad=True)


# Input is a tuple of tensors with shapes [2], [], [3,3,3] and dtypes int32, int64, float32 respectively
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, inputs):
        arg1, arg2, arg3 = inputs
        return torch.logical_or(arg1, arg2, out=arg3)

def my_model_function():
    return MyModel()

def GetInput():
    arg1 = torch.randint(-4, 16, (2,), dtype=torch.int32)
    arg2 = torch.randint(-16384, 32, (), dtype=torch.int64)
    arg3 = torch.rand((3, 3, 3), dtype=torch.float32)
    return (arg1, arg2, arg3)


# torch.rand(3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Create a parameter using _make_subclass to demonstrate the type issue
        data = torch.randn(3, dtype=torch.float32)
        self.param = torch.Tensor._make_subclass(nn.Parameter, data)

    def forward(self, x):
        return self.param + x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, dtype=torch.float32)


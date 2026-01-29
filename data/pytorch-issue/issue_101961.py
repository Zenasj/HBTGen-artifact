# torch.rand(1, 3, 1, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Create a parameter using torch.Tensor._make_subclass as per the example
        data = torch.randn(1, 3, 1, 1, dtype=torch.float32)
        self.param = torch.Tensor._make_subclass(nn.Parameter, data)  # Using two arguments as shown in the PR example
        
    def forward(self, x):
        return x + self.param  # Simple operation using the parameter

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 1, 1, dtype=torch.float32)


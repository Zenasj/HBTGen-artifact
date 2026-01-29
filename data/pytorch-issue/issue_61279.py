# torch.rand(10, 5, 10, dtype=torch.float32)  # Inferred input shape from example
import torch
from torch import nn
from functorch import vmap
import torch.nn.functional as F

class vmapTensor(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func == F.linear:
            # Apply vmap over the first dimension for both inputs and output
            func = vmap(func, (0, 0), 0)
        return super().__torch_function__(func, types, args, kwargs)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize weight as vmapTensor wrapped in a Parameter
        self.weight = nn.Parameter(vmapTensor(torch.randn(10, 5, 10)))
    
    def forward(self, x):
        return F.linear(x, self.weight)

def my_model_function():
    return MyModel()

def GetInput():
    # Return a vmapTensor instance matching the expected input shape
    return vmapTensor(torch.rand(10, 5, 10, dtype=torch.float32))


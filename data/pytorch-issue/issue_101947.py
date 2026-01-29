# torch.rand(10, dtype=torch.float16)
import torch
import torch.nn.functional as F
from torch import nn

class MyModel(nn.Module):
    def __init__(self, op, kwargs):
        super().__init__()
        self.operator = op
        self.kwargs = kwargs

    def forward(self, *args):
        return self.operator(*args, **self.kwargs)

def my_model_function():
    # Initialize with F.elu and fixed alpha=1.0 to avoid scalar input issues
    return MyModel(F.elu, {"alpha": 1.0})

def GetInput():
    # Return a random float16 tensor matching the model's expected input
    return torch.rand(10, dtype=torch.float16)


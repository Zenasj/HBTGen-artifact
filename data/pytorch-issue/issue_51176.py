# torch.rand(1, 3, 224, 224, dtype=torch.float32)
import torch
from torch import nn

def pass_through(a: int, b: int) -> tuple:
    return (a, b)

class MyModel(nn.Module):
    def __init__(self, a: int, b: int):
        super(MyModel, self).__init__()
        # Problematic tuple unpacking in __init__ for JIT
        self.a, self.b = pass_through(a, b)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.a + self.b

def my_model_function():
    return MyModel(a=1, b=2)  # Example initialization values

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)


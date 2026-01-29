# torch.rand(1, dtype=torch.float32)
import torch
from functorch.experimental.control_flow import cond
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell1 = nn.Parameter(torch.rand(3, 3))
        self.cell2 = nn.Parameter(torch.rand(3, 3))

    def forward(self, x):
        def then_func():
            return self.cell1
        def else_func():
            return self.cell2
        # Reproduces the Dynamo error when compiled
        return cond(x > 0, then_func, else_func, [])  # Output spec matches empty list in original issue

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.float32)


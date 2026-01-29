# torch.rand(4, dtype=torch.uint16)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a):
        # Division by torch.tensor(3) triggers dtype handling in export
        return a / torch.tensor(3)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random uint16 tensor matching the input shape (4 elements)
    return torch.randint(0, 256, (4,), dtype=torch.uint16)


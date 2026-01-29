# torch.rand(1, 1, 2, 2, dtype=torch.float32)  # Inferred from the 2x2 tensor in the distributed example
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return x.neg()  # Matches the operation in the distributed example's "a.neg()"

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 2, 2, dtype=torch.float32)  # Matches 2x2 input structure from the example


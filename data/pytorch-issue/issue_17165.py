# torch.rand(5, 10, dtype=torch.float), torch.rand(5, 10, dtype=torch.float)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(10, 5)
    
    def forward(self, x):
        a, b = x  # Unpack tuple of two tensors
        out1 = self.linear1(a)
        out2 = self.linear2(b)
        return out1 + out2

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.rand(5, 10), torch.rand(5, 10))  # Tuple of two tensors matching forward() requirements


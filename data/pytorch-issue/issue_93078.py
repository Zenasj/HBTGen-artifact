# torch.rand((), dtype=torch.bool)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, input):
        stacked = torch.dstack([input, input, input])
        return torch.reciprocal(stacked)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.tensor(True, dtype=torch.bool)


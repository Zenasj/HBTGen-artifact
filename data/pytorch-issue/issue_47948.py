# torch.rand((), dtype=torch.int32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return x  # Dummy model that returns input unchanged

def my_model_function():
    return MyModel()

def GetInput():
    # Generates a scalar integer tensor matching the test case's requirements
    return torch.randint(-5, 6, (), dtype=torch.int32)


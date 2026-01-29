# torch.randint(0, 100, (), dtype=torch.int64)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x + x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(low=0, high=100, size=())  # Returns a random 0-D integer tensor


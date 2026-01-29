# torch.rand(B, C, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(128, 128)
        
    def forward(self, x):
        return self.linear(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(10, 128)


# torch.rand(1, dtype=torch.float16)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        return torch.exp(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.float16)


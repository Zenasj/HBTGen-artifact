# torch.rand(41, dtype=torch.complex64)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, *args):
        neg = torch.neg(args[0])
        add = torch.add(args[0], args[0])
        return (neg, add)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(41, dtype=torch.complex64)


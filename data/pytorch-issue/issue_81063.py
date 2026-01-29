# torch.rand(8, 8), torch.rand(8)  # Input shapes for x and y
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs):
        x, y = inputs  # Unpack the tuple input
        x = torch.nn.functional.relu(x)
        z1 = x + y
        z2 = torch.sum(z1, 0)
        return z1, z2

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.randn(8, 8), torch.randn(8))  # Returns a tuple matching the model's input requirements


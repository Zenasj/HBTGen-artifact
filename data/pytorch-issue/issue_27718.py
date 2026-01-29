# torch.rand((), dtype=torch.float32)  # Input is a scalar float tensor
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return torch.arange(x)  # Uses float input to trigger ONNX ConstantOfShape type error

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a scalar float tensor as required by torch.arange
    return torch.rand((), dtype=torch.float32)


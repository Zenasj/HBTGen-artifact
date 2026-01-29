# torch.rand(10, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, y):
        return torch.asinh(y)

def my_model_function():
    return MyModel()

def GetInput():
    # Specific input tensor that triggers the discrepancy between eager/compiled modes
    y = torch.tensor([
        487875.875, -956238.8125, 630736.0, -161079.578125, 
        104060.9375, 757224.3125, -153601.859375, -648042.5, 
        733955.4375, -214764.90625
    ], dtype=torch.float32)
    return y


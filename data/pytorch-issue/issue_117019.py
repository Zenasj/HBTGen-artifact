# torch.rand(8, 6, 8, 6, 6, 1, dtype=torch.float64)

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return torch.diag_embed(input=input, dim1=-1, dim2=0, offset=1)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(8, 6, 8, 6, 6, 1, dtype=torch.float64)


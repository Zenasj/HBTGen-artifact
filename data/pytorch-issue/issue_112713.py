# torch.rand(6, 8, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.diag_embed(x, dim1=-1, dim2=0, offset=6)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(6, 8, dtype=torch.float32)


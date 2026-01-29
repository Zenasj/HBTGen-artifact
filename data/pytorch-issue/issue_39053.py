# torch.rand(3, 4, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def forward(self, x):
        return F.normalize(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, 4, dtype=torch.float32)


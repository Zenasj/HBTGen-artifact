# torch.rand(2, 8, 7, 10, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def forward(self, x):
        return F.interpolate(x, size=[1, 1], mode="bilinear")

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 8, 7, 10, dtype=torch.float32)


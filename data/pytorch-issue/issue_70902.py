# torch.rand(10, 3, 1000, 1000, dtype=torch.float32)
import torch
from torch import nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def forward(self, x: torch.Tensor):
        return F.interpolate(x, 2, mode="bilinear", align_corners=False)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(10, 3, 1000, 1000)


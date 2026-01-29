# torch.rand(B=2, C=5, H=3, W=4, dtype=torch.float32)  # Inferred from the example input shape
import torch
from torch import nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def forward(self, x):
        return F.unfold(x, kernel_size=(2, 3))

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 5, 3, 4, dtype=torch.float32)


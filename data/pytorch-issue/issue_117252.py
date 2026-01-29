# torch.rand(1, 1, 1, 1, 1, dtype=torch.int32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.fft.fft2(x, None, [], None)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint((1 << 15) - 1, [1,1,1,1,1], dtype=torch.int32)


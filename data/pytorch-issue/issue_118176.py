# torch.rand(4, 1, 3, 2, 8, 2, 6, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # Replicate the issue's core operation using the input tensor
        return torch.special.sinc(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand([4, 1, 3, 2, 8, 2, 6], dtype=torch.float32)


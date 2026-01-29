# torch.rand(B, C, H, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        return x.unfold(-1, 4, 2)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 24, dtype=torch.float32)


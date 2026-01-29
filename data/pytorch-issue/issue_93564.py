# torch.rand(B, C, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        if torch.distributed.is_available():
            return x.cos() + 1
        return 1

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(3, 4)


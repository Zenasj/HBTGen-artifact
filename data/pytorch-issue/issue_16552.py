# torch.rand(10, 2, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        return x  # Mimics the behavior of the C++ function returning the input tensor

def my_model_function():
    return MyModel()

def GetInput():
    return torch.ones((10, 2), requires_grad=True, device="cpu") * 2


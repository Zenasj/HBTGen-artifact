# torch.rand(2, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        a, b = x.tolist()
        return x + a + b

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, dtype=torch.float32)


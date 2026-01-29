# torch.rand(4, dtype=torch.float32)  # Inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, inp=None):
        return inp

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4)


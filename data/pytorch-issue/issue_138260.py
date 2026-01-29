# torch.rand(65, dtype=torch.float32, device="cuda")
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x=None):
        if x is None:
            x = torch.zeros(65, device="cuda")
        return torch.sin(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(65, dtype=torch.float32, device="cuda")


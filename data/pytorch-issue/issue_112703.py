# torch.rand(8, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        new_input = torch.rand(8, dtype=torch.float32, device=x.device)
        return torch.asinh(new_input, out=x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(8, dtype=torch.float32)


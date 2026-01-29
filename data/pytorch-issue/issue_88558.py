# torch.rand(B, 784, dtype=torch.float32)
import torch
import torch.nn as nn

class CustomSiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.silu_builtin = nn.SiLU()  # F.silu() implementation
        self.silu_custom = CustomSiLU()  # x * torch.sigmoid(x) implementation

    def forward(self, x):
        out_builtin = self.silu_builtin(x)
        out_custom = self.silu_custom(x)
        return torch.abs(out_builtin - out_custom).sum()  # Return sum of absolute differences

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 784, dtype=torch.float32)


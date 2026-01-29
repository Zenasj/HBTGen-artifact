# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        return torch.log(x)

def my_model_function():
    return MyModel()

def GetInput():
    B, C, H, W = 1, 1, 1, 1  # Inferred input shape based on issue's example
    x = torch.rand(B, C, H, W, dtype=torch.float32) * 2 - 1.0  # Generate values in [-1, 1] to test negative domain
    x.requires_grad_(True)
    return x


# torch.rand(N, dtype=torch.float32)
import torch
import torch.nn as nn

def diag2(x):
    diag_matrix = x.unsqueeze(1) * torch.eye(len(x), device=x.device)
    return diag_matrix

class MyModel(nn.Module):
    def forward(self, x):
        return diag2(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Example with N=5 (arbitrary length; adjust as needed)
    return torch.rand(5, dtype=torch.float32)


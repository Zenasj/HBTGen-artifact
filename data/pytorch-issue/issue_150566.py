# torch.rand(22, dtype=torch.int32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        shift = 8
        n = x.size(0)
        # Compute M4-style output (zeros beyond shifted range)
        m4 = torch.zeros_like(x)
        m4[:n - shift] = x[shift:]
        # Compute M1-style output (original values beyond shifted range)
        m1 = x.clone()
        m1[:n - shift] = x[shift:]
        # Return boolean indicating difference between M4 and M1 outputs
        return torch.any(m4 != m1)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.arange(22, dtype=torch.int32)


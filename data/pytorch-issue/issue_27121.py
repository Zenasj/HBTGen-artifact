# torch.rand(N, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Apply sinh and cosh operations (vectorized in the PR context)
        sinh_out = torch.sinh(x)
        cosh_out = torch.cosh(x)
        return (sinh_out, cosh_out)

def my_model_function():
    # Returns the model instance exercising vectorized sinh/cosh
    return MyModel()

def GetInput():
    # Generates input matching the benchmark's tensor shape and dtype
    return torch.rand(10000, dtype=torch.float32)


# torch.rand(128, 256, dtype=torch.bfloat16)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        return x * x

def my_model_function():
    # Initialize model with CUDA and bfloat16 as in the test case
    return MyModel().to(device='cuda', dtype=torch.bfloat16)

def GetInput():
    # Generate input matching CUDA device and bfloat16 dtype
    return torch.randn(128, 256, device='cuda', dtype=torch.bfloat16, requires_grad=True)


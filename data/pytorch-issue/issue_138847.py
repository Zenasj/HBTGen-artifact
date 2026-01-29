# torch.rand(B, dtype=torch.float, device="cuda")
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        if x.size(0) % 2 == 0:
            return x.clone() * 2
        else:
            return x.clone() * 0

def my_model_function():
    # Returns model on CUDA as in the original example
    return MyModel().to("cuda")

def GetInput():
    # Returns a 1D tensor of size 4 on CUDA (matches original input shape)
    return torch.rand(size=(4,), device="cuda")


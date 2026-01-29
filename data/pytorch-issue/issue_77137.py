# torch.rand(B, 5, dtype=torch.float32, device='cuda')
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(5, dtype=torch.float32))

    def forward(self, x):
        return self.weight + x

def my_model_function():
    # Initialize model on CUDA to replicate the issue scenario
    return MyModel().cuda()

def GetInput():
    # Match the input shape and device from the issue's example
    return torch.rand(2, 5, dtype=torch.float32, device='cuda')


# torch.rand(1, dtype=torch.float32, device='cuda:0')
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Reproduces the bug: integer tensor parameter on CUDA causes sharing error
        self.i = nn.Parameter(torch.tensor(0), requires_grad=False)
    
    def forward(self, x):
        # Dummy forward to satisfy module interface (input not used in original issue)
        return self.i  # Returns the problematic parameter

def my_model_function():
    # Returns the model instance with the faulty parameter configuration
    return MyModel()

def GetInput():
    # Generates a dummy input matching the expected shape (scalar)
    return torch.rand(1, dtype=torch.float32, device='cuda:0')


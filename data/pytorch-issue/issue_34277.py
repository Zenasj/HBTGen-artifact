# torch.rand(B, C, H, W, dtype=torch.float)  # Example input shape: (1, 1, 1, 1)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Ensure input is treated as float for Poisson distribution compatibility
        rate = x.float()  # Fix from the issue: Poisson requires float input
        dist = torch.distributions.Poisson(rate)
        return dist.sample()

def my_model_function():
    return MyModel()

def GetInput():
    # Return 4D tensor matching expected input shape (B, C, H, W)
    return torch.rand(1, 1, 1, 1, dtype=torch.float)


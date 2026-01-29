# torch.rand((), dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.x = nn.Parameter(torch.ones(10))  # Matches original x=torch.ones(10)
        self.x.register_post_accumulate_grad_hook(self.grad_hook)
    
    def grad_hook(self, param):
        # Replicates param.grad *= 2 from original hook
        param.grad *= 2
    
    def forward(self, input):
        # Forward pass: x * input (scalar multiplication)
        return self.x * input

def my_model_function():
    # Initialize model with correct parameter setup
    model = MyModel()
    return model

def GetInput():
    # Returns scalar tensor as input (matches original code's integer input)
    return torch.rand((), dtype=torch.float32)


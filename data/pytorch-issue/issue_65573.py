# torch.rand(B, C, H, W, dtype=torch.float32)  # Inferred input shape: 4D tensor with C=5 channels
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Example operation involving list mutation that triggered the JIT optimization bug
        indices = [0, 1, 3, 4]  # Original indices
        indices[-2] = 4  # Mutation causing symbolic shape analysis issue
        # Use indices to select channels (assuming input shape B,C,H,W)
        return x[:, indices]  # Accessing the 2nd dimension (channels)

def my_model_function():
    # Returns the model instance with default initialization
    return MyModel()

def GetInput():
    # Returns random input tensor matching expected dimensions
    return torch.rand(1, 5, 4, 4, dtype=torch.float32)  # B=1, C=5, H=4, W=4


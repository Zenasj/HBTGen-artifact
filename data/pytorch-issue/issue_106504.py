# torch.rand(1, 4096, dtype=torch.half)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, input_dim=4096, output_dim=4096):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Initialize shared weight matrix in half-precision
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim, dtype=torch.half))

    def forward(self, x):
        # Compute full linear layer output
        full_output = F.linear(x, self.weight)
        
        # Split weight matrix into two halves and compute outputs
        half_dim = self.output_dim // 2
        half1 = F.linear(x, self.weight[:half_dim])
        half2 = F.linear(x, self.weight[half_dim:])
        split_output = torch.cat([half1, half2], dim=1)
        
        # Return maximum absolute difference between the two methods
        return (full_output - split_output).abs().max()

def my_model_function():
    # Returns model initialized with default dimensions (4096 input/output)
    return MyModel()

def GetInput():
    # Returns random input tensor matching expected dimensions and dtype
    return torch.randn(1, 4096, dtype=torch.half)


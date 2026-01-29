# torch.rand(1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Fused both Linear configurations (with/without bias) as submodules
        self.linear_with_bias = nn.Linear(1, 1)
        self.linear_no_bias = nn.Linear(1, 1, bias=False)
    
    def forward(self, x):
        # Process input through both models and return outputs for comparison
        out_with_bias = self.linear_with_bias(x)
        out_no_bias = self.linear_no_bias(x)
        return (out_with_bias, out_no_bias)

def my_model_function():
    # Returns the fused model containing both Linear configurations
    return MyModel()

def GetInput():
    # Returns valid 1D input tensor (shape [1]) that works with both submodels
    return torch.randn(1)


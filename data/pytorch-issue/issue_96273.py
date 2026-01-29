# torch.rand(8, dtype=torch.float32)  # Input shape inferred as a 1D tensor of length 8
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.in0 = nn.Parameter(torch.tensor([0.0]))  # Scalar parameter for tmp0 initialization

    def forward(self, in1):
        tmp0 = self.in0[0].long()  # Convert to integer for indexing
        indices = torch.arange(in1.size(0), device=in1.device) + 5 * tmp0
        out0 = torch.zeros_like(in1)
        tmp = out0[indices]  # Load before store to replicate Triton's problematic order
        out0[indices] = in1  # Store operation
        out1 = tmp  # Capture pre-store values
        return out0, out1  # Return both outputs as per kernel's logic

def my_model_function():
    return MyModel()  # Initialize model with default parameters

def GetInput():
    return torch.rand(8, dtype=torch.float32)  # Matches size_hints=[8] from the issue


# torch.rand(16, dtype=torch.float, device="cuda")  # Input shape inferred from the repro's 'vals'
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.boundaries = torch.linspace(-1.0, 1.0, 33, device="cuda")
        # Create non-contiguous view of boundaries to trigger the bug
        self.register_buffer('noncontiguous_boundaries', self.boundaries[::2])

    def forward(self, vals):
        # Compare eager vs compiled results of bucketize on non-contiguous boundaries
        eager_result = torch.bucketize(vals, self.noncontiguous_boundaries)
        compiled_result = torch.compile(torch.bucketize)(vals, self.noncontiguous_boundaries)
        return torch.all(eager_result == compiled_result)

def my_model_function():
    # Returns the fused comparison model
    return MyModel()

def GetInput():
    # Generate random input matching the expected shape (16 elements on CUDA)
    return torch.randn(16, device="cuda")


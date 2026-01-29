import torch
import torch.nn as nn

# torch.rand(1, dtype=torch.int64)  # Input is a tensor containing window length N as integer
class MyModel(nn.Module):
    def __init__(self, periodic=True, dtype=None):
        super(MyModel, self).__init__()
        self.periodic = periodic
        self.dtype = dtype

    def forward(self, x):
        # Extract window length from input tensor
        N = x.item()
        # Generate Hann window using ONNX-supported operator
        return torch.hann_window(N, periodic=self.periodic, dtype=self.dtype)

def my_model_function():
    # Default parameters matching test case in PR (periodic=True, dtype=None)
    return MyModel(periodic=True)

def GetInput():
    # Return tensor with window length 10 (common test value)
    return torch.tensor([10], dtype=torch.int64)


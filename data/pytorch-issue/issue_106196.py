# torch.rand(B, T, C, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, C):
        super(MyModel, self).__init__()
        self.c_attn = nn.Linear(C, 4 * C, bias=True)  # Matches the Linear layer in the issue

    def forward(self, x):
        # Compute full output
        full_output = self.c_attn(x)
        # Slice input to last time-step and compute
        sliced_input = x[:, [-1], :]
        sliced_output = self.c_attn(sliced_input)
        # Extract corresponding slice from full output
        target_slice = full_output[:, [-1], :]
        # Compare using torch.allclose with tolerance (as per PyTorch documentation)
        result = torch.allclose(target_slice, sliced_output, atol=1e-7, rtol=1e-5)
        return torch.tensor(result, dtype=torch.bool)  # Return boolean result as tensor

def my_model_function():
    return MyModel(C=3)  # C=3 from the original example dimensions

def GetInput():
    B, T, C = 1, 2, 3  # Dimensions from the original issue's example
    return torch.randn(B, T, C)  # Matches the input generation in the issue


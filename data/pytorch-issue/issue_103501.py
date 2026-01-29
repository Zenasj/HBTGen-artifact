# torch.rand(1, 1, 2147483650, dtype=torch.float32)  # Inferred input shape
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, input):
        # Reproduces the problematic call to _softmax_backward_data
        return torch._softmax_backward_data(
            grad_output=input,  # Gradient of the output
            output=input,      # Output tensor from softmax
            dim=2,             # Dimension along which softmax is computed
            input_dtype=input.dtype  # Dtype for the result
        )

def my_model_function():
    # Returns an instance of MyModel without parameters
    return MyModel()

def GetInput():
    # Generates a CUDA tensor with the problematic large dimension
    return torch.rand(1, 1, 2147483650, dtype=torch.float32, device='cuda')


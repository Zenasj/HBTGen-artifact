# torch.rand(B, C, H, W, dtype=torch.float32)  # Inferred input shape: (B, C, H, W)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Apply torch.xlogy to the input tensor
        output = torch.xlogy(x, x)
        return output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # The input tensor should not contain negative values to avoid NaNs in the logarithmic operation
    B, C, H, W = 1, 3, 8, 8
    return torch.rand(B, C, H, W, dtype=torch.float32)


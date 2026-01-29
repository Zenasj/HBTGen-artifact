# torch.rand(B, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, input_tensor):
        size = int(input_tensor[0].item())  # Extract size as integer
        fill_value = input_tensor[1].item()  # Extract fill value
        # Generate tensors using both functions with consistent size format
        zeros = torch.zeros(size)  # torch.zeros accepts integer directly
        full = torch.full((size,), fill_value)  # torch.full requires tuple
        # Return difference to indicate semantic equivalence when fill_value is 0
        return full - zeros

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random input: [size (float), fill_value (float)]
    size = torch.randint(1, 5, (1,)).float()  # Random size between 1-4
    fill_value = torch.rand(1)                # Random fill value
    return torch.cat([size, fill_value])


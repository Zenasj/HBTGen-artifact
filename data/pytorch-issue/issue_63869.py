# torch.rand(B, C, H, W, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define any necessary layers or operations here
        # For this example, we will just use an identity layer
        self.identity = nn.Identity()

    def forward(self, x):
        # Perform the max operation along the specified dimension
        max_values, _ = x.max(dim=2)
        return max_values

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # The input shape is (B, C, H, W) where B is batch size, C is channels, H is height, and W is width
    # Using a smaller input to avoid OOM issues on typical GPUs
    B, C, H, W = 100, 1000, 3, 256
    return torch.rand(B, C, H, W, dtype=torch.float32).cuda()


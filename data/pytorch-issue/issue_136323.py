# torch.rand(B, N, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        # Cast the input tensor to int and replace inf with a large integer value
        x = x.to(torch.int)
        x[x == float('inf')] = int(1e6)
        
        # Call torch.unique on the modified tensor
        unique_values, counts = torch.unique(x, sorted=True, return_counts=True, dim=-1)
        return unique_values, counts

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Generate a random tensor input that matches the input expected by MyModel
    B, N = 4, 5  # Example batch size and number of elements
    x = torch.rand(B, N, dtype=torch.float32)
    x[torch.rand(B, N) > 0.5] = float('inf')  # Randomly set some elements to inf
    return x


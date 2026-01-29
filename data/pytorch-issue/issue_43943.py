# torch.rand(B, C, H, W, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define any necessary layers or components here
        # For this example, no additional layers are needed

    def forward(self, x):
        # Use index_select to slice multiple dimensions with sequences
        idx = torch.tensor([0, 1])
        x = x[:, :, idx]
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B, C, H, W = 1, 3, 10, 10  # Example dimensions
    return torch.rand(B, C, H, W, dtype=torch.float32)


# torch.rand(B, C, H, W, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(in_features=1024, out_features=64, bias=False)

    def forward(self, x):
        # Ensure the input is of type float32
        if x.dtype != torch.float32:
            raise ValueError("Input tensor must be of type torch.float32")
        return self.linear(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # The input shape is (B, 1024) where B is the batch size
    B = 11  # Example batch size
    input_tensor = torch.rand(B, 1024, dtype=torch.float32)
    return input_tensor


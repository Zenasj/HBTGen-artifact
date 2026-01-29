# torch.rand(B, C, H, W, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define any necessary layers or operations here
        self.softmax = nn.Softmax(dim=0)
        self.log_softmax = nn.LogSoftmax(dim=0)

    def forward(self, x):
        # Ensure the input tensor is of type float32
        x = x.to(torch.float32)
        # Apply softmax and log_softmax
        softmax_output = self.softmax(x)
        log_softmax_output = self.log_softmax(x)
        return softmax_output, log_softmax_output

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Assuming a 1D tensor for simplicity
    B, C, H, W = 1, 1, 1, 9  # Example dimensions
    return torch.rand(B * C * H * W, dtype=torch.float32)

# Example usage:
# model = my_model_function()
# input_tensor = GetInput()
# softmax_output, log_softmax_output = model(input_tensor)


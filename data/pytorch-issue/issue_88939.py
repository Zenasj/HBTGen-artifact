# torch.rand(1, 10, 540, 540, dtype=torch.bfloat16) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # No additional parameters or layers needed for this model

    def forward(self, x):
        # Perform the interpolation
        return torch.nn.functional.interpolate(x, 2, mode='bilinear', align_corners=True)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 10, 540, 540, dtype=torch.bfloat16)


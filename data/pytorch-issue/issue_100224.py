# torch.rand(1, 3, 2, 2, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        out = x
        mask1 = out > 0
        out[mask1] *= 0.5
        return out

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    torch.manual_seed(420)
    return torch.randn(1, 3, 2, 2)


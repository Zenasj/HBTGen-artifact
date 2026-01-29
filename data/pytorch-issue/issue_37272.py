# torch.rand(1, 4, 4, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        # Compute the norm along dimensions 1 and 2, keeping the dimensions
        return torch.norm(x, dim=[1, 2], keepdim=True)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(1, 4, 4, dtype=torch.float32).cuda()


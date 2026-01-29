# torch.rand(B, C, H, W, dtype=torch.bfloat16)  # Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return torch.nan_to_num(x, nan=0.0, posinf=torch.finfo(torch.bfloat16).max, neginf=torch.finfo(torch.bfloat16).min)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B, C, H, W = 1, 1, 1, 1
    return torch.rand(B, C, H, W, dtype=torch.bfloat16)


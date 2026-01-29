# torch.rand(B, C, H, W, dtype=...)  # Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        # Generate an affine grid
        grid = F.affine_grid(x, torch.Size([x.size(0), 1, 28, 28]))
        return grid

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # The input shape is (B, 2, 3) where B is the batch size
    B = 64
    x = torch.rand(B, 2, 3, dtype=torch.float32, device='cuda')
    return x


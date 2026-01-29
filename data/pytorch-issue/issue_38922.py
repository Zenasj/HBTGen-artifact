# torch.rand(B, C, H, W, dtype=...)  # The input shape is (B, 900, 1) where B can vary, e.g., (40, 900, 1)

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        # Perform the argmax operation along the specified dimension
        return x.argmax(dim=2)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # The shape is (40, 900, 1) as per the issue description
    return torch.full((40, 900, 1), 0, dtype=torch.float32)


# torch.rand(1, 2, 376, 16, 16, dtype=torch.float32)  # Inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Use the correct floor division to avoid the warning
        return torch.div(torch.tensor([x.shape[0]], device=x.device), 4, rounding_mode='floor')

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 2, 376, 16, 16, dtype=torch.float32)


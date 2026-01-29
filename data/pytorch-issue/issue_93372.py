# torch.rand(1, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = torch.nn.Linear(in_features=1, out_features=2)
    
    def forward(self, x):
        # Ensure the input has at least 2 dimensions
        if x.dim() == 1:
            x = x.unsqueeze(0)
        y = self.layer0(x)
        return y

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, dtype=torch.float32)


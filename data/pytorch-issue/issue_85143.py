# torch.rand(1, 2, 3, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    def forward(self, x):
        x = x.to(self.device)
        out = F.pad(x, [0, 0, 0, 1], "constant", value=0.0)
        return out

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 2, 3, dtype=torch.float32)


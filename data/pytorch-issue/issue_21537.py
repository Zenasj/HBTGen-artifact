# torch.rand(1, 1, 10, 10, dtype=torch.float32)  # Inferred input shape

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(1, 1, 1)
        self.lin = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.lin(x))
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.ones((1, 1, 10, 10))


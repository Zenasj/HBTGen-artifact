# torch.rand(B, 1, 28, 28, dtype=torch.float32)  # Inferred input shape: (batch_size, channels, height, width)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    batch_size = 64
    return torch.rand(batch_size, 1, 28, 28, dtype=torch.float32)


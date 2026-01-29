# torch.rand(N, C, L, dtype=torch.float32)  # Inferred input shape: (batch_size, channels, length)

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.bn = nn.BatchNorm1d(4)  # C = 4 from the issue

    def forward(self, x):
        return self.bn(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    N, C, L = 2, 4, 5  # Batch size, Channels, Length
    return torch.rand((N, C, L), dtype=torch.float32)


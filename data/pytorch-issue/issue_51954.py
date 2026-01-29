# torch.rand(B, C, H)  # Inferred input shape: (batch_size, channels, height)

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(768, 768)
        self.max_pool_1d = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.linear(x)
        x = x.unsqueeze(0)  # Add a dimension to make it 3D
        x = self.max_pool_1d(x)
        x = x.squeeze(0)  # Remove the added dimension
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(32, 768)


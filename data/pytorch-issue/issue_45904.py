# torch.rand(B, C, L, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.maxpool = nn.MaxPool1d(2, stride=1, return_indices=True)

    def forward(self, x):
        # MaxPool1d with return_indices=True returns a tuple (output, indices)
        output, indices = self.maxpool(x)
        return output, indices

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B, C, L = 1, 1, 10  # Batch size, Channels, Length
    return torch.rand(B, C, L, dtype=torch.float32)


# torch.rand(B, C, H, W, dtype=...)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dims):
        super(MyModel, self).__init__()
        self.layers = []

        prev_dim = input_size
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.ReLU(True))
            prev_dim = hidden_dim
        self.layers.append(nn.Linear(prev_dim, output_size))

        self.layer_module = nn.Sequential(*self.layers)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    input_size = 2
    output_size = 2
    hidden_dims = [128] * 3  # Example hidden dimensions
    return MyModel(input_size, output_size, hidden_dims)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    batch_size = 10
    input_size = 2
    return torch.rand(batch_size, input_size)


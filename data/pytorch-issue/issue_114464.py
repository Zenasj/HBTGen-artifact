# torch.rand(1, 4, dtype=torch.int64)  # Inferred input shape based on the example

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        y = x.clone().reshape(-1, 4)
        y[:, [2, 0]] = y[:, [0, 2]]
        return y

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.tensor([0, 1, 2, 3], dtype=torch.int64)


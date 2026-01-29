# torch.tensor(20, dtype=torch.int64)  # Inferred input shape: scalar tensor

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        return torch.tensor([x], dtype=torch.int64)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.tensor(20, dtype=torch.int64)


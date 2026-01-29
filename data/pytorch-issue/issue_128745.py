# torch.rand(B, 4, 4, dtype=torch.float32)  # Assuming B is the batch size and the input shape is (B, 4)

import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor
import torch.distributed as dist

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = nn.Sequential(*[nn.Linear(4, 4, bias=False) for _ in range(2)])

    def forward(self, x):
        return self.model(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B = 1  # Assuming a batch size of 1 for simplicity
    return torch.rand(B, 4, 4, dtype=torch.float32)


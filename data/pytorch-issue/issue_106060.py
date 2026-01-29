# torch.rand(4, dtype=torch.float32, device="cuda")  # Inferred input shape

import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.register_buffer("buffer", torch.tensor([0.5, 0.5, 0.5, 0.5], device="cuda"))

    def forward(self, x):
        # should be a no-op, but causes dynamo to lose the static input
        self.buffer = self.buffer.to(x)
        self.buffer.add_(x)
        return self.buffer + x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(4, device="cuda")


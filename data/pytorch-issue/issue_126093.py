# torch.rand(B, C, H, W, dtype=torch.float32)  # Assuming a typical input shape for a linear layer with quantization

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(4, 4)
        self.unary_fn = nn.ReLU()
        self.linear2 = nn.Linear(4, 4)
        self.unary_fn2 = nn.ReLU()

    def forward(self, x):
        x = self.unary_fn(self.linear(x))
        x = self.unary_fn2(self.linear2(x))
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn((2, 4), dtype=torch.float32)


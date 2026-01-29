# torch.rand(1, 1, 1, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.ZeroPad1d(padding=2)

    def forward(self, x):
        x = self.layer1(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand([1, 1, 1], dtype=torch.float32)


# torch.rand(10, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, return_duplicate):
        super(MyModel, self).__init__()
        self.return_duplicate = return_duplicate

    def forward(self, x):
        return x if not self.return_duplicate else (2 * x, x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel(return_duplicate=False)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(10)


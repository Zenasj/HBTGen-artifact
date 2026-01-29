# torch.rand(64, 2048, 64, dtype=torch.float16) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(64, 64, 192, dtype=torch.float16))
        self.bias = torch.nn.Parameter(torch.randn(64, 1, 192, dtype=torch.float16))

    def forward(self, x):
        return torch.ops.aten.baddbmm.default(self.bias, x, self.weight)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(64, 2048, 64, dtype=torch.float16)


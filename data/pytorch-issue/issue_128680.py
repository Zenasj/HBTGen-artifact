# torch.rand(B, 10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(10, 10, bias=False)
        self.b = nn.Linear(10, 1, bias=False)

    def forward(self, x):
        a_out = self.a(x)
        b_out = self.b(x)
        return (a_out, b_out)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(10, 10)  # Matches the input shape expected by MyModel


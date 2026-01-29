# torch.rand(B, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        code = compile("print('Hello World')", "foo", "exec")
        exec(code)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3)


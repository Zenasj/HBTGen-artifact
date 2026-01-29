# torch.rand(B, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, input):
        n = input.size(-1)
        output = input + int(n * 0.2) + 1
        return output, input + 1

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(5)  # Matches the input shape used in the original repro example


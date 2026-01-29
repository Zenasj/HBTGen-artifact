# Inputs: (torch.rand(8, 6, 8, 1, 6, 1, 3, 4, 1, 3, dtype=torch.float16), torch.rand(8, 6, 8, 1, 6, 1, 3, 4, 1, 3, dtype=torch.float32), torch.rand(8, 6, 8, 1, 6, 1, 3, 4, 1, 3, dtype=torch.float32))
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs):
        out, input, other = inputs
        out = torch.ge(out=out, input=input, other=other)
        return out

def my_model_function():
    return MyModel()

def GetInput():
    input1 = torch.rand(8, 6, 8, 1, 6, 1, 3, 4, 1, 3, dtype=torch.float16)
    input2 = torch.rand(8, 6, 8, 1, 6, 1, 3, 4, 1, 3, dtype=torch.float32)
    input3 = torch.rand(8, 6, 8, 1, 6, 1, 3, 4, 1, 3, dtype=torch.float32)
    return (input1, input2, input3)


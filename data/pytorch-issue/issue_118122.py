# Inputs: (torch.zeros(3,5, dtype=torch.bfloat16), torch.ones(2,5, dtype=torch.bfloat16), torch.tensor([[0,1,2,0,0]], dtype=torch.int64))
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        inp, src, index = inputs
        return inp.scatter_add(0, index, src)

def my_model_function():
    return MyModel()

def GetInput():
    dtype = torch.bfloat16
    inp = torch.zeros(3, 5, dtype=dtype)
    src = torch.ones(2, 5, dtype=dtype)
    index = torch.tensor([[0, 1, 2, 0, 0]], dtype=torch.int64)
    return (inp, src, index)


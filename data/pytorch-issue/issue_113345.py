# torch.rand(20, dtype=torch.float32), torch.rand(10, dtype=torch.float32)  # input shape
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs):
        primals, tangents = inputs
        primals_1, primals_2 = primals.split([10, 10])
        tangents_1 = tangents

        abs_1 = torch.abs(primals_1)
        add = abs_1 + 1
        div = primals_1 / add

        sum_1 = torch.sum(primals_2)
        lt = sum_1 < 0  # boolean scalar

        neg = -tangents_1
        div_1 = primals_1 / add  # redundant but preserved per original code
        div_2 = div_1 / add
        mul = neg * div_2
        div_3 = tangents_1 / add
        sign = torch.sign(primals_1)
        mul_1 = mul * sign
        add_1 = div_3 + mul_1

        return (div, lt, add_1)

def my_model_function():
    return MyModel()

def GetInput():
    primals = torch.rand(20, dtype=torch.float32)
    tangents = torch.rand(10, dtype=torch.float32)
    return (primals, tangents)


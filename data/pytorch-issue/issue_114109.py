# torch.rand(1, 32, dtype=torch.int64), torch.rand(1, 32), torch.zeros(1, 32), torch.rand(1, 32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        arg0_1, arg3_1, convert_element_type, convert_element_type_1 = inputs
        scatter_add = convert_element_type.scatter_add(0, arg0_1, convert_element_type_1)
        div = arg3_1 / scatter_add
        return div

def my_model_function():
    return MyModel()

def GetInput():
    arg0_1 = torch.randint(0, 1, (1, 32), dtype=torch.int64)
    arg3_1 = torch.randn(1, 32)
    convert_element_type = torch.zeros(1, 32)
    convert_element_type_1 = torch.randn(1, 32)
    return (arg0_1, arg3_1, convert_element_type, convert_element_type_1)


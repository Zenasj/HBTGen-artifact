# (torch.rand(5), torch.rand(5))
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs):
        param, param2 = inputs
        tensor_list = {param2}
        return torch.tensor([param in tensor_list], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.rand(5), torch.rand(5))


# torch.rand(2, 2, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        return x.clone(memory_format=torch.preserve_format)

def my_model_function():
    return MyModel()

def GetInput():
    # Create expand tensor with strides (1, 0)
    base = torch.randn([2, 1])
    arg = base.expand(2, 2)
    return arg


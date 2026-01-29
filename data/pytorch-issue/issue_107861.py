# torch.rand(4, 3, 4, 4, dtype=torch.float32)
import torch
from torch import nn

def traverse_obj(obj, func):
    if isinstance(obj, (tuple, list)):
        a = type(obj)(traverse_obj(o, func) for o in obj)
        return a
    elif isinstance(obj, dict):
        return {name: traverse_obj(o, func) for name, o in obj.items()}
    elif isinstance(obj, (torch.Tensor, torch.nn.Parameter)):
        return func(obj)
    else:
        return obj

class MyModel(nn.Module):
    def __init__(self, mem_format=torch.channels_last):
        super().__init__()
        self.mem_format = mem_format

    def forward(self, inputs):
        def inner_to_mem_format(obj):
            old_requires_grad = obj.requires_grad
            _tensor = obj.clone().to(memory_format=self.mem_format).detach()
            _tensor.requires_grad_(old_requires_grad)
            return _tensor
        return traverse_obj(inputs, inner_to_mem_format)

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.randn(4, 3, 4, 4).requires_grad_(True),)


import torch.nn as nn

import torch
from torch.nn import Parameter
from typing import Callable

called = False
class BasevLLMParameter(Parameter):
    """
    Base parameter for vLLM linear layers. Extends the torch.nn.parameter
    by taking in a linear weight loader. Will copy the loaded weight
    into the parameter when the provided weight loader is called.
    """

    def __new__(cls, data: torch.Tensor, **kwargs):

        return super().__new__(cls, data=data, requires_grad=False)

    def __init__(self, data: torch.Tensor, weight_loader: Callable):
        self._weight_loader = weight_loader

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        global called
        called = True
        if kwargs is None:
            kwargs = {}
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)


model = torch.nn.Linear(3, 3)
model.weight = BasevLLMParameter(model.weight, weight_loader=lambda x: x)
model.bias = BasevLLMParameter(model.bias, weight_loader=lambda x: x)

@torch.compile(fullgraph=True)
def f(x):
    y = model(x)
    return y

x = torch.randn(2, 3)
f(x)
print(called)
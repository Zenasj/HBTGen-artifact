# torch.rand(2), torch.rand(2)  # Input is a named tuple of two tensors of shape (2,)
import torch
from collections import namedtuple

def isinstance_namedtuple(obj) -> bool:
    return (
            isinstance(obj, tuple) and
            hasattr(obj, '_asdict') and
            hasattr(obj, '_fields')
    )

class MyModel(torch.nn.Module):
    def forward(self, inputs):
        if isinstance_namedtuple(inputs):
            return type(inputs)(*(torch.mul(x, 2) for x in inputs))
        else:
            return type(inputs)([torch.mul(x, 2) for x in inputs])

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.rand(2)
    y = torch.rand(2)
    MyTuple = namedtuple('MyNamedTuple', ['foo', 'bar'])
    return MyTuple(foo=x, bar=y)


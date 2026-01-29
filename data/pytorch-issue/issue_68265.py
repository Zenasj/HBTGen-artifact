# torch.rand(2, dtype=torch.float32)
import torch
from torch import nn

class WrapperTensor(torch.Tensor):
    elem: torch.Tensor

    __slots__ = ['elem']

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        r = torch.Tensor._make_wrapper_subclass(
            cls, elem.size(),
            dtype=elem.dtype, layout=elem.layout,
            device=elem.device, requires_grad=elem.requires_grad,
            strides=elem.stride(), storage_offset=elem.storage_offset()
        )
        r.elem = elem
        return r

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        new_args = []
        for a in args:
            if isinstance(a, cls):
                new_args.append(a.elem)
            else:
                new_args.append(a)
        return func(*new_args, **(kwargs or {}))

class MyModel(nn.Module):
    def forward(self, x):
        return WrapperTensor(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(2, 2)[:, 1]


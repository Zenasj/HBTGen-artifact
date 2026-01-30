import torch
from torch.utils._python_dispatch import TorchDispatchMode

class Foo(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args, kwargs=None):
        return func(*args, **(kwargs or {}))


class MyTensor(torch.Tensor):
    def __new__(cls, data: torch.Tensor):
        return data.as_subclass(cls)


t1 = torch.rand(10, requires_grad=True)
t2 = t1 + t1
m1 = MyTensor(t2)
with Foo():
    m2 = MyTensor(t2)
import torch
import abc
import torch.fx._symbolic_trace

class Meta(abc.ABCMeta, torch.fx._symbolic_trace.ProxyableClassMeta):
    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)
        x.attr = 100
        return x

class Multistreamable(abc.ABC):
    pass

class Foo(Multistreamable, metaclass=Meta):
    pass

@torch.compile(backend="eager", fullgraph=True)
def f(x):
    typ = type(Foo())
    typ.__bases__
    return x + 1

f(torch.randn(1))
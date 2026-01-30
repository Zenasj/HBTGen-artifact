import torch
import torch.nn as nn

class InnerModule(torch.nn.Module):

    def forward(self, t):
        return t + t

class MyModule(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, t):
        x = self.inner(t)
        y = self.inner(t)
        return x + y

class MyTracer(torch.fx.Tracer):
    def is_leaf_module(self, module, name):
        return True

cls = type(self)
cls_call = cls.__call__
...
def wrapped_call(self, *args, **kwargs):
    try:
        return cls_call(self, *args, **kwargs)
    except Exception as e:
        ...
cls.__call__ = wrapped_call
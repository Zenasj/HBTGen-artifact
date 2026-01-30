import torch

# simple passthrough TorchDispatchMode
class CustomDispatchMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        return func(*args, **kwargs)

# derive from TwoTensor to minimize boilerplate
class MySubclass(TwoTensor):
    def __torch_dispatch__(self, func, types, args, kwargs=None):
        with torch.overrides.enable_reentrant_dispatch():
            return func(args[0].a)

t = MySubclass(torch.rand(2), torch.rand(2))
with CustomDispatchMode():
    res = t.clone()
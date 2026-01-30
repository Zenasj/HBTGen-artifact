import torch
import torch.nn as nn

class TestMode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                print(func._schema.name)
                print(args)
                out = func(*args, **kwargs)
                print(args)
                return out

a = torch.rand((3,3))
with enable_torch_dispatch_mode(TestMode()):
        torch.nn.functional.rrelu(a, training=True)
import torch
import weakref

class UserDefined:
    def __init__(self, x):
        self.x = x

    def __call__(self):
        return self.x.sin()

def f(obj):
    return ref()

obj = UserDefined(torch.randn(3))

ref = weakref.ref(obj)
ret = torch.compile(f, backend="eager", fullgraph=True)(ref)
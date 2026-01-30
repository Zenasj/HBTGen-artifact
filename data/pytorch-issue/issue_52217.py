import torch
import torch.nn as nn

def get_overload_annotations(mod):
    # original function => [(mangled overload name, overload function)]
    overloads = {}

    for name in dir(type(mod)):
        item = getattr(mod, name, None)
        if not callable(item):
            continue

class A(nn.Module):
    __jit_ignored_attributes__ = ["ignored"]
    def __init__(self):
        super().__init__()
    @property
    def ignored(self):
        raise ValueError("shouldn't be called")

torch.jit.script(A()) # will error
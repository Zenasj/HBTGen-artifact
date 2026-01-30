import collections
import torch

class Obj(collections.OrderedDict):
    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

def fn(obj, x):
    obj["asdf"] = x + 1
    return x

opt_fn = torch.compile(fn, backend="eager")
obj = Obj()
inp = torch.randn(3, 3)
opt_fn(obj, inp)

import collections
from dataclasses import dataclass, fields
import torch

class Base(collections.OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __post_init__(self):
        class_fields = fields(self)
        for field in class_fields:
            v = getattr(self, field.name)
            if v is not None:
                self[field.name] = v

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

@dataclass
class Derived(Base):
    field: torch.Tensor

@torch._dynamo.disable(recursive=False)
def fn(x):
    return Derived(x)

opt_fn = torch.compile(fn, backend="eager")
inp = torch.randn(3, 3)
opt_fn(inp)
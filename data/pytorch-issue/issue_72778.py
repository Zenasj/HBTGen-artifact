import torch

from collections import OrderedDict
from torch.nn import Linear

m = Linear(1, 1)
print(m.state_dict())  # works
print(m.state_dict(OrderedDict()))  # fails

from torch.nn import Linear

m = Linear(1, 1)
print(m.state_dict())  # works
print(m.state_dict('a'))  # fails

def _state_dict_impl(self, destination, prefix, keep_vars):
    ...  # real things here

def state_dict(self, prefix, keep_vars):
    return self._state_dict_impl(None, prefix, keep_vars)
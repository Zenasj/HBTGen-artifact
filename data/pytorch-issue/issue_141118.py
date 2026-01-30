import torch.nn as nn

import torch
from tensordict import TensorDict

td = TensorDict(a=1, b=2, c=True)

@torch.compile(fullgraph=True)
def add1(td):
    return TensorDict(**td)+1

add1(td)

import torch
from tensordict import TensorDictParams, TensorDict

td = TensorDictParams(TensorDict(a=1, b=2, c=True))

@torch.compile(fullgraph=True)
def add1(td):
    return TensorDict(**td)+1

add1(td)

import torch
from torch import nn
import collections

# class MyWeirdDict(collections.abc.MutableMapping):  # Works
class MyWeirdDict(collections.abc.MutableMapping, nn.Module):  # breaks
    def __init__(self, **kwargs):
        super().__init__()
        self._items = kwargs
    def keys(self):
        return self._items.keys()
    def __getitem__(self, item):
        return self._items[item]
    def __setitem__(self, key, value):
        self._items[key] = value
    def __delitem__(self, item):
        del self._items[item]
    def __len__(self):
        return len(self._items)
    def __iter__(self):
        yield from self._items
    def __hash__(self):
        return hash(id(self))
    def items(self):
        for k, v in self._items.items():
            yield (k, v)

@torch.compile(fullgraph=True)
def to_weird_dict(td):
    return MyWeirdDict(**td)

d = MyWeirdDict(a=1, b=2, c=3)
to_weird_dict(d)
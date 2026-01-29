import torch
from torch import nn
from collections.abc import MutableMapping

# torch.rand(2, 3)  # Input shape inferred from MyWeirdDict's initialization
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = MyWeirdDict(a=torch.randn(2, 3), b=torch.randn(2, 3))

    def forward(self, x):
        # Operation that triggers Dynamo error when compiled
        new_dict = MyWeirdDict(**self.params)  # Unpacking nn.Module+MutableMapping
        return x + 1  # Dummy output to satisfy forward contract

class MyWeirdDict(MutableMapping, nn.Module):
    def __init__(self, **kwargs):
        # Explicitly initialize both parent classes
        MutableMapping.__init__(self)
        nn.Module.__init__(self)
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

def my_model_function():
    # Returns an instance with properly initialized modules
    return MyModel()

def GetInput():
    # Matches the input shape expected by MyModel's forward()
    return torch.rand(2, 3)


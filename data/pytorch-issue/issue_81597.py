# torch.rand(B, 3, 32, 32, dtype=torch.float32)
import torch
from torch import nn
from weakref import WeakKeyDictionary

class WeakTensorRefKey:
    def __init__(self, tensor):
        self.tensor = tensor
        self._id = id(tensor)

    def __hash__(self):
        return hash(self._id)

    def __eq__(self, other):
        if not isinstance(other, WeakTensorRefKey):
            return False
        return self._id == other._id

class WeakTensorKeyDictionary:
    def __init__(self):
        self._dict = WeakKeyDictionary()

    def __setitem__(self, key, value):
        wrapped_key = WeakTensorRefKey(key)
        self._dict[wrapped_key] = value

    def __getitem__(self, key):
        wrapped_key = WeakTensorRefKey(key)
        return self._dict[wrapped_key]

    def __contains__(self, key):
        wrapped_key = WeakTensorRefKey(key)
        return wrapped_key in self._dict

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cache = WeakTensorKeyDictionary()

    def forward(self, x):
        if x in self.cache:
            return self.cache[x]
        else:
            # Example computation: double the input tensor
            result = x * 2
            self.cache[x] = result
            return result

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random input tensor with shape (1, 3, 32, 32)
    return torch.rand(1, 3, 32, 32, dtype=torch.float32)


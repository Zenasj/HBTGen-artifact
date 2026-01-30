import torch

if __name__ == "__main__":
    t = torch.tensor([1, 2, 3], dtype=torch.float64)

try:
    import torch.tensor as tensor
    print(type(tensor))
except Exception as error:
    print(error)

try:
    from torch import tensor
    print(type(tensor))
except Exception as error:
    print(error)

import sys
import warnings

class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target = None):
        if fullname == "torch.tensor":
            warnings.warn("You are importing wrongly!")


sys.meta_path.insert(0, _Finder())

import sys


class _Modules(dict):
    def __getitem__(self, item):
        if item == "torch.tensor":
            warnings.warn("You are importing wrongly!")
        return super().__getitem__(item)


sys.modules = _Modules(sys.modules)

import torch
import sys
import importlib

from torch import tensor
print(tensor)

import torch.tensor as tensor
print(tensor)

print(getattr(sys.modules["torch"], "tensor"))

print(sys.modules["torch.tensor"])
print(importlib.import_module("torch.tensor"))
print(__import__("torch.tensor", globals(), locals(), [None], 0))
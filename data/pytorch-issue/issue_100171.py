import torch

py
torch.ops.load_library("somelib.so")
c = torch.classes.somelib.SomeClass()
print(len(c))
# raise NotImplementedError

py
torch.ops.load_library("somelib.so")
c = torch.classes.somelib.SomeClass()
print(len(c))
# raise NotImplementedError: '__len__' is not implemented for __torch__.torch.classes.somelib.SomeClass
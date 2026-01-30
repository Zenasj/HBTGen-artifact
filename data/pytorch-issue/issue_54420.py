import torch
import gc

print(torch.__version__)

a = torch.rand(10)

a = gc.get_objects()
print("All ok")

import torch
import gc

print(torch.__version__)

a = torch.rand(10)

print(gc.get_objects())

import torch

print(torch.__dict__)
print(torch.classes)

from torch._classes import _Classes
print(_Classes())
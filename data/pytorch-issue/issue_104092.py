# Manual repro
import torch
from torch import _dynamo
torch._dynamo.config.verbose=True
print(torch.__version__)
class HasCustomIndexing():
  def __init__(self):
    self.l = list(range(5))

  def __getitem__(self, item):
    return self.l[item]

def regular_indexing():
  return HasCustomIndexing()[1]

def tensor_indexing():
  return HasCustomIndexing()[torch.as_tensor([1])]

print("Regular indexing")
print(regular_indexing()) # 1
print(torch.compile(regular_indexing, backend='eager')()) # 1
print("Tensor indexing")
print(tensor_indexing()) # 1
print(torch.compile(tensor_indexing, backend='eager')()) # crash

import torch
def fun():
  return list(range(5))[torch.as_tensor([1])]
print(fun()) #1
torch.compile(fun)() #Error
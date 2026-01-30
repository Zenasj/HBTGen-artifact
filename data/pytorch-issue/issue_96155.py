import torch

def fn(a):
     b = a[[0]]
     return b

a = torch.tensor([])
fn(a)
# IndexError: index is out of bounds for dimension with size 0
torch.compile(fn)(a)
# segfault
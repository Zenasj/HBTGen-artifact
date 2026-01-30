py
import torch

a = torch.tensor([])

def func(a):
    b = a[[0]]
    return b

torch.compile(func)(a)
# segmentation fault (core dumped)
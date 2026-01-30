import torch

from torch import _dynamo as dy

def func():
    a = torch.logspace(3, 10 + 3j, steps=5)


func()
opt_func = dy.optimize('eager')(func)
opt_func()

aten.logspace
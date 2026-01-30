import torch

from torch import _dynamo as dyna

def func():
    return torch.normal(1, 1, (8,8))

op = dyna.optimize('eager')(func)
a = func() # this runs fine
a = op() # this fails
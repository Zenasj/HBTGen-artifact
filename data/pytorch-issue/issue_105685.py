import torch
from torch import _dynamo as dynamo

@dynamo.optimize('eager')
def func():
    t = torch.rand(3,4)
    t.copy_(3)
    print(t)

func()

import torch

def func():
    t = torch.rand(3,4)
    t.copy_(3)
    print(t)

func()

tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]])
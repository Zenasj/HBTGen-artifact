import torch
from tensordict import TensorDict

@torch.compile(fullgraph=True)
def func(x):
    if hasattr(x, "to"):
        return x.to("cpu")
    return x
func(torch.randn(3))

import torch
from tensordict import TensorDict

@torch.compile(fullgraph=True)
def func(x):
    if hasattr(x, "hocuspocus"):
        return x.to("cpu")
    return x
func(torch.randn(3))
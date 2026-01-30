import torch

@torch.jit.script
def fn(x):
    # type: (List[int])
    return tuple(x)

@torch.jit.script
def fn(x):
    # type: (List[int])
    return (x,)
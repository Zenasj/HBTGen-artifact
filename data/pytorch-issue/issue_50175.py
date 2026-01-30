import torch

@torch.jit._overload
def fn(x: int) -> int:
    pass

@torch.jit._overload
def fn(x: str) -> str:
    pass

def fn(x):
    if isinstance(x, int):
        return x + 3
    else:
        return x

@torch.jit._overload
def fn(x: int) -> int:
    return x + 3

@torch.jit._overload
def fn(x: str) -> str:
    return x
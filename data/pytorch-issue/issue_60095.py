import torch
from typing import List

def fn(x: int):
    if torch.jit.isinstance(x, (List[str], str)):
        z = x + "bar"
        return z
    else:
        return "baz"
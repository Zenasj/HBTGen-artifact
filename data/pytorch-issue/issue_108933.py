import torch
from enum import Enum

class Color(int, Enum):
    RED = 1
    GREEN = 2

def enum_fn(x: Color, y: Color) -> bool:
    if x == Color.RED:
        return True
    return x == y

m = torch.jit.script(enum_fn)
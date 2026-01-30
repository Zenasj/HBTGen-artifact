import torch
from typing import Any

def fn(t: Any):
    if isinstance(t, tuple):
        a, b = t
        return a + b

torch.jit.script(fn)
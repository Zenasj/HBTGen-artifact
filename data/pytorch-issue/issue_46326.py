import torch
from typing import NamedTuple
_MyNamedTuple = NamedTuple('_MyNamedTuple', [('value', int)])

@torch.jit.script
def foo():
    return _MyNamedTuple(1)
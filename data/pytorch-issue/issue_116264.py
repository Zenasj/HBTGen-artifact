import torch
from dataclasses import dataclass

@dataclass
class C:
    x: int


@torch.compile(backend="aot_eager", fullgraph=True)
def f():
    l = C(3)
    r = C(4)
    return l == r

f()
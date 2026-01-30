py
import torch
import dataclasses

@dataclasses.dataclass
class Base:
    x: int = 15.0
    y: int = 0


@torch.compile(fullgraph=True)
def fn(x, d):
    z = Base(15, 0)
    z.x += 2
    x.x += 1

    return x.x * d * z.x

x = Base(15, 0)

fn(x, torch.ones(2, 2))
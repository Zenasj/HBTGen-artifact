import torch
from dataclasses import dataclass, field
from typing import Any

@dataclass
class DClass:
   sharding_contexts: Any = field(default_factory=list)
   a: int = 1

def fn(c, x, inp_list):
    d = DClass(inp_list)
    d.sharding_contexts.append(x.sin())
    return d

c = DClass()
x = torch.randn(4)
inp_list = [torch.randn(4)]
opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
opt_ret = opt_fn(c, x, inp_list)
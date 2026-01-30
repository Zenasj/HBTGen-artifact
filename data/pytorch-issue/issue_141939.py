import torch.nn as nn

import traceback
import torch
from types import MethodType
from typing import Any, Dict
import torch.nn

class Block(torch.nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.to_out = torch.nn.ModuleList([])
        self.to_out.append(torch.nn.Linear(i, o, bias=True))
        self.to_out.append(torch.nn.Dropout(0.5))

    def forward(self, x):
        for l in self.to_out:
            x = l(x)
        return x

class Problem1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleDict({f"{i}": Block(64, 64) for i in range(2)})
    
    def forward(self, x):
        for k, m in self.blocks.items():
            x = m(x)
        return x

class Problem2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block(64, 64) for i in range(2)])
        
    def forward(self, x):
        x = self.blocks[0](x)
        for m in self.blocks[1:]:
            x = m(x)
        return x

def _split_after_forward(self, *args, **kwargs):
    return self._orig_forward(*args, **kwargs)

def annotate_split_points(mod: torch.nn.Module, spec: Dict[str, Any]):
    for qualname, split_type in spec.items():
        atoms = qualname.split(".")
        predecessor_module = mod
        for i, atom in enumerate(atoms[:-1]):
            try:
                predecessor_module = getattr(predecessor_module, atom)
            except AttributeError as e:
                raise e
        mod_to_wrap = getattr(predecessor_module, atoms[-1])
        mod_to_wrap._orig_forward = mod_to_wrap.forward
        mod_to_wrap.forward = MethodType(_split_after_forward, mod_to_wrap)

for problem in [Problem1, Problem2]:
    print("-" * 40 + f" {problem.__name__} " + "-" * 40)
    m = problem()
    m(torch.rand(64, 64))
    # simpified torch.distributed.pipeline code
    split_spec = {}
    for j in range(1):
        split_spec[f"blocks.{j * 2 + 1}"] = 1
    try:
        if not isinstance(m, Problem2):
            annotate_split_points(m, split_spec)
        gm = torch.export.export(m, (torch.rand(64, 64),))
        for n in gm.graph_module.graph.nodes:
            if "nn_module_stack" in n.meta and "slice(" in list(n.meta["nn_module_stack"].values())[-1][0]:
                print("WRONG MODULE STACK", n.meta["nn_module_stack"])
        torch.export.unflatten(gm)
    except Exception as e:
        traceback.print_exc()
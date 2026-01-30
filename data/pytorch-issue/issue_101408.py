import torch.nn as nn

import torch
from torch.fx import symbolic_trace
from functorch.compile import minifier


class Model(torch.nn.Module):
    def forward(self, x):
        x = torch.ops.aten.div(x, x)
        x = torch.ops.aten.add(x, 3)
        x = torch.ops.aten.mul(x, x)
        return torch.ops.aten.sub(x, x)

model = Model()
inps = torch.randn(3)

my_module_traced = symbolic_trace(model)

def pass_checker(fx_g, inps):
    return (torch.ops.aten.mul in set([i.target for i in fx_g.graph.nodes]))

min_f, inps = minifier(my_module_traced, inps, pass_checker)
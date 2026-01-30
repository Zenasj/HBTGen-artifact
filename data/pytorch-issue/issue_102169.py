import torch.nn as nn

import torch
from functorch import make_fx
from functorch.compile import minifier

class Model(torch.nn.Module):
    def forward(self, _to_copy_1):
        cumsum = torch.ops.aten.cumsum.default(_to_copy_1, 1);  _to_copy_1 = None
        return cumsum

model = Model()
inps = (torch.ones(torch.Size([1, 12]), dtype=torch.int32), )
out = model(*inps)

fx_g = make_fx(model)(*inps)

def fails(fx_g, inps):
    return (torch.ops.aten.cumsum.default in (i.target for i in fx_g.graph.nodes))

minifier(fx_g, inps, fails)
import torch
import torch.nn as nn

class M(torch.nn.Module):
    def forward(self, x):
        return torch.ops.aten.sum.dim_IntList(x, [])

ep = torch.export._trace._export(M(), (x,), strict=False, pre_dispatch=False)

print(torch._export.serde.serialize.serialize(ep).exported_program)
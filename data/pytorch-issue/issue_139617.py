import torch.nn as nn

import torch

class M(torch.nn.Module):
    def forward(self, x):
        return torch.full_like(x, 1)

ep = torch.export.export(M(), (torch.rand(3,3),))
print(ep) # aten::full_like in the graph
ep = ep.run_decompositions()
print(ep) # aten::full_like still in the graph

decomp = torch._decomp.get_decompositions([torch._ops.ops.aten.full_like]) # it is empty

exit()
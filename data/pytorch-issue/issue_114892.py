import torch.nn as nn

import torch


class SingleOp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.ops.aten.scatter_add

    def forward(self, t, dim, index, src, **kwargs):
        return self.op(t, dim, index, src, **kwargs)


t = torch.randn(10, 5)
dim = -1
index = torch.tensor([[2, 4, 3, 1, 0],[0, 2, 1, 4, 3],[3, 1, 4, 2, 0],[4, 0, 3, 1, 2],[3, 0, 4, 1, 2]])
src = torch.randn(5, 5)

model = SingleOp()
output = model(t, dim, index, src)
print("Success on torch")

ep = torch.export.export(model, args=(t, dim, index, src))
ep.run_decompositions(decomp_table=torch._decomp.decomposition_table)
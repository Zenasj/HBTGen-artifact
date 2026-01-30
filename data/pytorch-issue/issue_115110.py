import torch.nn as nn

import torch


class SingleOp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.ops.aten.full

    def forward(self, size, fill_value, **kwargs):
        return self.op(size, fill_value, **kwargs)


size = (10,)
fill_value = False

model = SingleOp()
output = model(size, fill_value)
print("Success on torch")

ep = torch.export.export(model, args=(size, fill_value))
# ep.run_decompositions(decomp_table=torch._decomp.decomposition_table)
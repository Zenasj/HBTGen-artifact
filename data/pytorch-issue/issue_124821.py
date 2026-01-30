import torch.nn as nn

import numpy as np
import torch

# Model definition
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v2_0 = torch.nn.Parameter(torch.empty([1, 22, 51], dtype=torch.int64), requires_grad=False)

    def forward(self, _args):
        v2_0 = self.v2_0
        getitem = _args
        max_1 = getitem.max(0)
        getattr_1 = max_1.values
        max_2 = torch.max(getitem, v2_0)
        return (getattr_1, max_2)

m = M()

inp =  torch.from_numpy(np.zeros((22, 51),dtype=np.int64))
m(inp) # this line is OK
opt = torch.compile(m, fullgraph=True, backend='inductor', mode=None)
opt(inp) # this line will crash
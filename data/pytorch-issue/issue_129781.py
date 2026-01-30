import torch.nn as nn

import torch
from torch._export.converter import TS2EPConverter

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.should_show = torch.nn.Linear(1,1)
        self.should_NOT_show = torch.nn.Linear(1,1)

    def forward(self, x):
        if x.size()[0] > 0:
            return self.should_show(x)
        else:
            return self.should_NOT_show(x)


inp = (torch.randn(1),)
ts_model = torch.jit.trace(M(), inp)
print(ts_model.graph)
print(ts_model.state_dict())

ep = TS2EPConverter(ts_model, inp).convert()
print(ep.state_dict())
import torch.nn as nn

import torch
from torch.nn import *

class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        arange_1 = torch.arange(512, -512, -1.0)
        return arange_1

mod = Repro()
opt_mod = torch.compile(mod)
print(opt_mod().dtype)
print(mod().dtype)

torch.int64
torch.float32
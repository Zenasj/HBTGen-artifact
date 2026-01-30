import torch.nn as nn

import torch
print(torch.__version__)
from torch import nn

class Mod(nn.Module):
    def __init__(self):
        super().__init__()
        self.foo = nn.ParameterList([nn.Parameter(torch.rand(10))])

mod = Mod()
mod.eval()
mod.train()
import torch.nn as nn

import torch
from typing import Optional, List

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, pkv: Optional[List[torch.Tensor]] = None, ql: Optional[torch.Tensor] = None):
        real_seq_length = x.size(1)
        if pkv is not None:
            real_seq_length += pkv[0].size(2) if ql is None else ql

torch.jit.script(Model())
import torch.nn as nn
import numpy as np

import torch
from torch.fx import symbolic_trace

class MyConv2d(torch.nn.Module):
    def __init__(self):
        super(MyConv2d, self).__init__()
        self.conv2d = torch.nn.Conv2d(64, 3, (3, 3))
        pass

    def forward(self, inp):
        if inp.is_contiguous(memory_format=torch.contiguous_format):
            inp = inp.to(memory_format=torch.channels_last)
        return self.conv2d(inp)

m = MyConv2d()
trace  = symbolic_trace(m)

torch.memory_format
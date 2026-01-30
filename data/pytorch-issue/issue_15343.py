import numpy as np

import torch
import torch.nn as nn

conv = nn.Conv2d(512, 16, 1, stride=0, padding=0, bias=False)
inp = torch.load('fpe_input.torchtensor')
assert inp.shape == torch.Size([1, 512, 26, 40])

# Range of the input values
assert inp.min().item() == 0.0
assert inp.max().item() < 27750

# No NaN values:
assert (inp == inp).sum() == inp.numel()

# FPE happens here:
out = conv(inp)
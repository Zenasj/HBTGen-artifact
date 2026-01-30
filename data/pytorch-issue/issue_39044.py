import torch.nn as nn
import numpy as np

import torch
import math
m = torch.nn.FractionalMaxPool2d(3, output_size=(4, 4))
inp=torch.randn(2, 2, 30, 30, device="cuda")
inp[0][0][0][0] = math.nan
print(m(inp))  # no nan in the input
inp.fill_(-math.inf)
print(m(inp)) # prints tensor filled with finfo.min, would assert if asserts were enabled
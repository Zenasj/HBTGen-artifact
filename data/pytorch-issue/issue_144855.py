import torch.nn as nn

import torch
x = torch.ones((1, 256, 16, 720, 1280), dtype=torch.bfloat16).cuda()
out = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
assert (out[0] == out[-1]).all()
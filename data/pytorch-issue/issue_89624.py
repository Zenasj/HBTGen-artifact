import torch.nn as nn

import torch
t = torch.rand(3, 214, 320)
# this works
torch.nn.functional.pad(t, tuple([0, 0, 0, 10, 0, 0]), value=0)
# and this does not:
torch.nn.functional.pad(t.to(torch.device('mps')), tuple([0, 0, 0, 10, 0, 0]), value=0)
# IndexError: Dimension out of range (expected to be in range of [-3, 2], but got 3)
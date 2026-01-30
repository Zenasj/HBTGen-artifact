import torch.nn as nn

import torch

size_in = 56  # the sizes here matter - these were the lowest ones that triggered the error, but many higher values also work
size_out = 444

layer = torch.nn.Linear(size_in, size_out)
t = torch.randn(size_in)

torch.set_num_threads(1)
out1 = layer(t)

torch.set_num_threads(6)
out2 = layer(t)

(out1 == out2).all(), abs(out1 - out2).max()
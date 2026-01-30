import torch.nn as nn

# Maxpool of non-positive values, such that padding is included in the computation of each result element.
import torch
r = torch.nn.functional.max_pool2d(-torch.rand(2,2,64), kernel_size=[3,3], padding=1)
assert torch.equal(r, torch.zeros_like(r))
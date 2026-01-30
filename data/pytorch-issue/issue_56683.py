import torch.nn as nn

import torch
import torch.nn.functional as F

a = torch.ones(1, 2) / 2
a.requires_grad_()
b = torch.ones(1, 2) / 2

matching_matrix_unused_for_backprop = F.binary_cross_entropy(a, b.requires_grad_(), reduce = 'none')

#  File ".../vadim/prefix/miniconda/lib/python3.8/site-packages/torch/nn/functional.py", line 2525, in binary_cross_entropy
#    return torch._C._nn.binary_cross_entropy(
# RuntimeError: the derivative for 'target' is not implemented
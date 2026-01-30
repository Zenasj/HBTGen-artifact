import torch.nn as nn

import torch
from torch.nn.functional import kl_div

input = torch.rand(()).log()
target = torch.tensor(-1, dtype=input.dtype)

print(kl_div(input, target))
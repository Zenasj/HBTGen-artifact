import torch.nn as nn

import torch

self = torch.full((2, 3, 5, 3, 7,), 1,
                  dtype=torch.float64, requires_grad=False)
output_size = [1879048192, 1879048192, 1879048192]
torch._C._nn.adaptive_avg_pool3d(self, output_size)
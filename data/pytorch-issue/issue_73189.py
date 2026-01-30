import torch.nn as nn

import torch

self = torch.full((2, 4, 3, 3,), 0.5, dtype=torch.float64, requires_grad=False)
kernel_size = [1879048192, 1879048192]
dilation = [1, 1]
padding = [1250999896764, 1250999896764]
stride = [1250999896764, 1250999896764]
torch._C._nn.im2col(self, kernel_size, dilation, padding, stride)
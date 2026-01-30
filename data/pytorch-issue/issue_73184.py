import torch.nn as nn

import torch

self = torch.full((2, 16, 4,), 0.5, dtype=torch.float64, requires_grad=False)
output_size = [536870912, 536870912]
kernel_size = [2, 2]
dilation = [536870912, 536870912]
padding = [1250999896764, 1250999896764]
stride = [1250999896764, 1250999896764]
torch._C._nn.col2im(self, output_size, kernel_size, dilation, padding, stride)
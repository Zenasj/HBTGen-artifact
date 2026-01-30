import torch.nn as nn

3
import torch
import torch.nn.functional as F
device = torch.device('mps')

input = torch.rand(2, 3, 16, 16, device=device)
kernel = torch.rand(1, 1, 3, 11, device=device)
tmp_kernel = kernel.expand(-1, 3, -1, -1)
output = F.conv2d(input, tmp_kernel, groups=1, padding=0, stride=1)
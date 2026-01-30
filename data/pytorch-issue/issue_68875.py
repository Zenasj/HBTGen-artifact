import torch.nn as nn

import torch
input = torch.rand(torch.Size([1, 12, 12]), dtype=torch.float32)
output_size = [4, 5]
kernel_size = [2, 2]
dilation = 8
padding = 0
stride = 1
res1 = torch.nn.functional.fold(input, output_size, kernel_size, dilation, padding, stride)
res2 = torch.nn.functional.unfold(res1, kernel_size, dilation, padding, stride)
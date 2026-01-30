import torch.nn as nn

import torch

input = torch.rand(2048, 256, 7, 7).cuda()
weight = torch.rand(4096, 9216).cuda()
bias = torch.rand(4096).cuda()

torch.nn.functional.linear(input, weight, bias)

#     return torch._C._nn.linear(input, weight, bias)
# RuntimeError: CUDA out of memory. Tried to allocate 56.00 GiB (GPU 0; 10.76 GiB total capacity; 242.02 MiB already allocated; # 9.51 GiB free; 244.00 MiB reserved in total by PyTorch)
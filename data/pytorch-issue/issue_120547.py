import torch.nn as nn

import torch
import torch.nn.functional as F

weight = torch.ones([16, 1, 32])
bias = torch.ones([16])
stride, padding, dilation, groups = (1, 16, 1, 16)

input_1 = torch.rand((1, 16, 1))

F.conv1d(input_1, weight, bias, stride, padding, dilation, groups)
print("input_1: ok")

input_2 = torch.rand((16, 1, 1))
input_2 = input_2.transpose(0, 1)

F.conv1d(input_2, weight, bias, stride, padding, dilation, groups)
print("input_2: ok")

# Case of RuntimeError: cannot create std::vector larger than max_size()
input_2 = torch.rand((1, 1, 16))
input_2 = input_2.transpose(1, 2)
# weight = torch.ones([16, 1, 16])  # the process freezes and loads 1 core at 100%
F.conv1d(input_2, weight, bias, stride, padding, dilation, groups)
print("input_3: ok")
import torch.nn as nn

import torch
from torch import nn
print(torch.backends.cudnn.is_acceptable(torch.cuda.FloatTensor(1)))

m = nn.Conv2d(8, 13, 3, stride=2).cuda()
input = torch.randn(5, 8, 20, 30, device="cuda")
output = m(input)
print("success", output.shape)

import torch.__config__
print(torch.__config__.show())
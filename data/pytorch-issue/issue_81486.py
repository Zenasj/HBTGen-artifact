import torch
import torch.nn as nn
bn = nn.BatchNorm2d(128).cuda()
x = torch.randn([256, 128, 294, 294], device='cuda')
x = bn(x)
print(x.shape)
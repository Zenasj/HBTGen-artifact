import torch.nn as nn

import torch
x=torch.rand(2048,256,8,8).cuda()
up=torch.nn.Upsample(scale_factor=2)
print(up(x).size())

import torch
x=torch.rand(2048,256,4,4).cuda()
up=torch.nn.Upsample(scale_factor=2)
print(up(x).size())  #output torch.Size([2048, 256, 8, 8])
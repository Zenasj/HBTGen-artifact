import torch
import torch.nn as nn

conv1=nn.Conv3d(1, 10, kernel_size=(3, 3, 10), stride=(1, 1, 1), padding=(1, 1, 0)).cuda()

x = torch.randn(24, 1, 100, 375, 128).cuda()
print(conv1(x))

x = torch.randn(25, 1, 100, 375, 128).cuda()
print(conv1(x))
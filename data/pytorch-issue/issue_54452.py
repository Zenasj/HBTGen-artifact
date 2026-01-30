import torch.nn as nn

import torch

x = torch.randn(4, 8, 32, 32, device='cuda', dtype=torch.float)
net = torch.nn.ConvTranspose2d(8, 16, kernel_size=3, padding=[3, 3])
net = net.cuda()

y = net(x)
torch.cuda.synchronize()
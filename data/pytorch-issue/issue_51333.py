import torch
from torch.nn import Conv3d

_BATCH_SIZE = 3
model = Conv3d(2048, 512, kernel_size=[3, 1, 1], stride=[1, 1, 1], padding=[1, 0, 0], bias=False).cuda()
x = torch.ones(_BATCH_SIZE, 2048, 4, 10, 13).cuda()
y = model(x)
print(y)
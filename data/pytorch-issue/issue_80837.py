import torch
import torch.nn as nn


conv = nn.Conv2d(
    1,
    128,
    kernel_size=(5, 2),
    stride=(2, 1),
    padding=(0, 1),
    dilation=(1, 1),
    groups=1,
    bias=True,
    padding_mode='zeros')

t = torch.rand([1, 2, 321, 201, 1])
t = torch.transpose(t, 1, 4)
t2 = t[..., 0]
r = conv(t2)
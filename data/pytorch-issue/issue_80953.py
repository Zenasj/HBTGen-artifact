import torch
import torch.nn as nn

N, C, D, H, W = 1, 32, 3, 6, 6
out_channels = C*32
kernel_size=2
stride=1
padding=2
dilation=1
groups=2

conv = nn.Conv3d(
    in_channels=C,
    out_channels=out_channels,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding,
    dilation=dilation,
    groups=groups,
)
x = torch.randn([N, C, D, H, W])

# normal forward/backward works fine
out = conv(x)
out.sum().backward()

#ExpandedWeights crashes with shape error
out = call_for_per_sample_grads(conv, x.shape[0], x)
out.sum().backward()
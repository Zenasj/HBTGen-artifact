import torch.nn as nn

import torch

ifm = torch.ones([1, 1, 2, 2]).contiguous(memory_format=torch.channels_last)
model = torch.nn.ConvTranspose2d(in_channels=1, out_channels=2, kernel_size=[5, 5])
model.weight.data = torch.ones([1, 2, 5, 5]).contiguous(
    memory_format=torch.channels_last
)
fwd_out = model(ifm)
print(fwd_out)
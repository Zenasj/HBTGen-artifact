import torch.nn as nn

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

conv2d_pt = torch.nn.Conv2d(
    128, 128,
    (7, 7),
    stride=(1, 1),
    padding=(0, 0), groups=1, bias=True
)

class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv2d = conv2d_pt

    def forward(self, x):
        return self.conv2d(x)

model = ConvNet()

torch._C._set_mkldnn_enabled(False)
x = torch.ones((64, 128, 256, 256))
for _ in range(100):
    model(x)
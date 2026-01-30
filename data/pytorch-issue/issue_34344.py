import torch.nn as nn

import torch
import torch.backends.mkldnn

def test_conv3d_backward():
    torch.manual_seed(2019)
    N = 64
    C = 1
    x = torch.randn(N, C, 16, 16, 16).requires_grad_()
    conv3d = torch.nn.Conv3d(in_channels=C,
                             out_channels=1,
                             kernel_size=1,
                             stride=2,
                             padding=1,
                             bias=False,
                             groups=1)
    with torch.backends.mkldnn.flags(enabled=False):
        conv3d(x).sum().backward()
    print(conv3d.weight.grad)

for i in range(20):
    test_conv3d_backward()
import torch.nn as nn

import copy
import torch
from torch.utils import mkldnn as mkldnn_utils
import torch.testing

dtype=torch.bfloat16
torch.manual_seed(1234)
for _ in range(12):
    N = torch.randint(1, 3, (1,)).item()
    M = torch.randint(1, 3, (1,)).item()
    C = torch.randint(1, 3, (1,)).item()
    x_shape = (N, C) + (224,)
    x = torch.randn(x_shape, dtype=torch.float32)
    conv = torch.nn.ConvTranspose1d(in_channels=C,
                        out_channels=M,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        dilation=2,
                        bias=True,
                        groups=1).float()
    x_lower = x.to(dtype=dtype)
    mkldnn_conv = mkldnn_utils.to_mkldnn(copy.deepcopy(conv))
    mkldnn_conv_lower = mkldnn_utils.to_mkldnn(copy.deepcopy(conv), dtype)
    y = mkldnn_conv(x.to_mkldnn()).to_dense()
    y_lower = mkldnn_conv_lower(x_lower.to_mkldnn()).to_dense(torch.float32)
    torch.testing.assert_close(y, y_lower, atol=1e-1, rtol=1e-3)
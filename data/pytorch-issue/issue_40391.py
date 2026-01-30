import torch.nn as nn

import torch
import copy
from torch.utils import mkldnn as mkldnn_utils
groups=1
N = 32
C = 32
M = 32
x = torch.randn(N, C, 56, 56, dtype=torch.float).to_mkldnn()
conv2d = torch.nn.Conv2d(in_channels=C,
                            out_channels=M,
                            kernel_size=3,
                            padding=1)
mkldnn_conv2d = mkldnn_utils.to_mkldnn(copy.deepcopy(conv2d))
with torch.backends.mkldnn.flags(enabled=False):
    y_mkldnn = mkldnn_conv2d(x).to_dense()
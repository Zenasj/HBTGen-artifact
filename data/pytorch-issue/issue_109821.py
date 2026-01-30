import torch.nn as nn

import torch
import torch._lazy
import torch_mlir._mlir_libs._REFERENCE_LAZY_BACKEND as lazy_backend

# on CPU device
img_cpu = torch.zeros([2,3,10,10,10])
out_cpu = torch.nn.AdaptiveAvgPool3d(2)(img_cpu)

# on LTC device
lazy_backend._initialize()
device = 'lazy'
img_lazy = img_cpu.to(device)
out_lazy = torch.nn.AdaptiveAvgPool3d(2).to(device)(img_lazy)
torch._lazy.mark_step()

# This check should be True, but it is False
assert out_lazy.numel() == out_cpu.numel(), f"Got {out_lazy.shape} shape on LTC, but {out_cpu.shape} on CPU device"
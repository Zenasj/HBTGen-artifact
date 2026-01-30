py
from lazy_tensor_core import debug
import torch
import lazy_tensor_core
import lazy_tensor_core.debug.metrics as metrics

lazy_tensor_core._LAZYC._ltc_init_ts_backend()

device = 'lazy'
dtype = torch.float32

mat = torch.randn(2, 3, device=device, dtype=dtype)
vec = torch.randn(3, device=device, dtype=dtype)
print(torch.mv(mat, vec))

print(metrics.metrics_report())
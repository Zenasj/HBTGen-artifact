py
from lazy_tensor_core import debug
import torch
import lazy_tensor_core
import lazy_tensor_core.debug.metrics as metrics

lazy_tensor_core._LAZYC._ltc_init_ts_backend()

torch.manual_seed(42)

device = 'lazy'
dtype = torch.float32

x = torch.randn(3, 3, device=device, dtype=dtype)
y = torch.randn(3, 3, device=device, dtype=dtype)

print(torch.mm(x, y))
print(metrics.metrics_report())
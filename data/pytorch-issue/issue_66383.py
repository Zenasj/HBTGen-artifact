import torch
import lazy_tensor_core as ltc
from lazy_tensor_core.debug import metrics

ltc._LAZYC._ltc_init_ts_backend()
device = 'lazy'
dtype = torch.float32

x = torch.randn((2,3,4), device=device, dtype=dtype)
y = torch.randn((2,3,4), device=device, dtype=dtype)

def computation(x,y):
    # clamp with scalar bounds
    a = torch.clamp(x, 0.2, 1.0)
    # clamp with tensor bounds
    b = torch.clamp(a, torch.tensor(0.5, device=device))
    # clamp with out tensor specified
    c = torch.clamp(b, 0.1, out=y)
    return c

computation(x, y)
print(metrics.metrics_report())
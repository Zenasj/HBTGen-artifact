import torch
import lazy_tensor_core as ltc
from lazy_tensor_core.debug import metrics

ltc._LAZYC._ltc_init_ts_backend()
device = 'lazy'
dtype = torch.float32

x = torch.randn((2,3,4), device=device, dtype=dtype)

def computation(x):
    return x.sigmoid() * x.rsqrt()

computation(x)
print(metrics.metrics_report())
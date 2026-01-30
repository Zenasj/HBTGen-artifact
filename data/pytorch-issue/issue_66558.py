import torch
import lazy_tensor_core as ltc
from lazy_tensor_core.debug import metrics

ltc._LAZYC._ltc_init_ts_backend()
device = 'lazy'
dtype = torch.int

x = torch.ones((2,3,4), device=device, dtype=dtype)
y = torch.zeros((2,3,4), device=device, dtype=dtype)

def computation(x,y):
    return torch.bitwise_and(x, torch.bitwise_and(y, 1))

print(computation(x, y))
print(metrics.metrics_report())
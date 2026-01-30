import torch.nn as nn

import torch
import lazy_tensor_core
import lazy_tensor_core.debug.metrics

lazy_tensor_core._LAZYC._ltc_init_ts_backend()
device = 'lazy'

inp = torch.rand((3, 3, 100, 100), device=device)
target = torch.zeros((3, 100, 100), dtype=torch.long, device=device)

print(torch.nn.functional.nll_loss(inp, target))
print(lazy_tensor_core.debug.metrics.metrics_report())
from timeit import default_timer

import torch
import torch.utils.cpp_extension
from torch.utils.benchmark import Measurement, Timer

t = Timer(
    stmt=f"y.copy_(x);torch.mps.synchronize()",
    setup=f"x=torch.rand(4, 5, 16, 64, 33, 24, dtype=torch.float32, device='mps')[:,:,:,:24,:24,];y=torch.empty(x.shape, device=x.device, dtype=x.dtype)",
    language="python", timer=default_timer
)
print(t.blocked_autorange())
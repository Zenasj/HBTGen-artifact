py
#!/usr/bin/env python3

import torch

src = torch.empty((0, 2, 3), dtype=torch.float16)  # 0 elems
self = torch.empty(1, dtype=torch.complex64)  # 1 elem

print(src.numel())
print(self.numel())

self.real = src # XXX: trigger bug
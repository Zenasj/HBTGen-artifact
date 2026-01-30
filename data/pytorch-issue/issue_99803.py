import torch
from torch.utils.dlpack import to_dlpack, from_dlpack

a = torch.randint(0, 100, (1, 77), dtype=torch.int32)
b = from_dlpack(to_dlpack(a))

print(a.stride())
print(b.stride())

(77, 1)
(1, 1)
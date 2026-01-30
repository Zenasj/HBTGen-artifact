import torch
a = torch.rand(1200, 1000, 3, 256).cuda() # about 3.4GiB
b = a.max(dim=2)

import torch
a = torch.rand(2300, 1000, 3, 256).cuda() # about 6.6GiB
b = a.max(dim=2)
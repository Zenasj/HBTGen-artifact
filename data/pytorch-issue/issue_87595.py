import torch
a = torch.tensor([2, 2, 3]).cuda(0)
print(a.prod())

import torch
a = torch.tensor([2, 2, 3]).cuda(1)
print(a.prod())
import torch

a = torch.Tensor()              # OK
b = torch.Tensor([1, 2, 3])     # inspection triggered
c = torch.Tensor(12)            # inspection triggered
d = torch.Tensor(a)             # inspection triggered

import torch

a = torch.Tensor()              # OK
b = torch.Tensor([1, 2, 3])     # inspection triggered
c = torch.Tensor(12)            # inspection triggered
d = torch.Tensor(a)             # inspection triggered
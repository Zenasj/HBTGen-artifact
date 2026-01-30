import torch

torch.manual_seed(1)
x = torch.randint(0, 5, (1000, ))

x.eq(x).sum()

import torch as t
a = t.tensor([1,2,3,4])
a.sum()
3
import torch
a=torch.ones(1,2,2, device=torch.device('mps'))
print (a)
print (a.transpose(2,1))

import torch
x = torch.rand((10, 5, 4), device=torch.device("mps"))
x.transpose(0, 1)
import torch
from torch import linalg

my_tensor = torch.tensor([-3., -2., -1., 0., 1., 2.])

print(linalg.norm(input=my_tensor, ord=-1)) # tensor(0.)
print(torch.min(torch.sum(torch.abs(input=my_tensor), dim=0))) # tensor(9.)

import torch

torch.__version__ # '2.3.0'
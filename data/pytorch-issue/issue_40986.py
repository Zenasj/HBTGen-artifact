import torch

a=torch.tensor(2.)
b=torch.tensor(2., device="cuda")
a += b
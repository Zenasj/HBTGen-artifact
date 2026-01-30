import torch
from numpy import linalg as nla

n = 100
a = torch.rand(n, n)
H = a @ a.t() + torch.diag_embed(torch.rand(n))

n = 100
a = torch.rand(n, n)
H = a @ a.t() + torch.diag_embed(-10 + torch.rand(n))

M = torch.rand(n, n)
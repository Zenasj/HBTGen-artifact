python
import torch

def norm_except_dim(v, pow, dim):
    return torch.norm_except_dim(v, pow, dim)

compiled_model = torch.compile(norm_except_dim)

pow = 1
dim = 0

v1 = torch.rand(2)
r1 = compiled_model(v1, pow, dim)
print(r1)

v2 = torch.rand(5)
r2 = compiled_model(v2, pow, dim)
print(r2)
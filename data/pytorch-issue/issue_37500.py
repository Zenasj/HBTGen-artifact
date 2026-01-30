py
import torch

x = torch.randn(2 ** 20)
with torch.autograd.profiler.profile() as prof:
    torch.einsum("i->", x)

print(prof)
print(prof.table())
print(prof)
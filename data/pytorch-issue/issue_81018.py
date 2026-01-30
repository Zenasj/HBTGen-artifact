import torch

mask = torch.ones(5, dtype=torch.bool, device='cuda')
s = torch.tensor(float('inf'))
val = torch.randn(5, device='cuda')

torch.masked_fill(val, mask, s)
# tensor([inf, inf, inf, inf, inf], device='cuda:0')

torch._refs.masked_fill(val, mask, s)
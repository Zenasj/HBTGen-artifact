import torch
x = torch.tensor([1], dtype=torch.float64, device='cuda')
exp = torch.tensor([128], dtype=torch.int64, device='cuda')

torch.ldexp(x, exp)

tensor([inf], device='cuda:0', dtype=torch.float64)

torch.tensor([2**128], dtype=torch.float64, device='cuda')

tensor([3.4028e+38], device='cuda:0', dtype=torch.float64)
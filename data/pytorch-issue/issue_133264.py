import torch
x = torch.tensor([1], dtype=torch.float16, device='cuda')
exp = torch.tensor([1], dtype=torch.int8, device='cuda')

torch.ldexp(x, exp).dtype
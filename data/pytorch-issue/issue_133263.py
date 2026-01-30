import torch
x = torch.tensor([1], dtype=torch.bfloat16, device='cuda')
torch.frexp(x)
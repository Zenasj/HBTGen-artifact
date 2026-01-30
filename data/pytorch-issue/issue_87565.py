py
import torch
print(torch.linalg.cond(torch.ones(5, 5, device='cuda'))) # tensor(6.9682e+15, device='cuda:0')
print(torch.linalg.cond(torch.ones(5, 5, device='cpu'))) # tensor(inf)
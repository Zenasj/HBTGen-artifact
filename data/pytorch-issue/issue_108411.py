import torch

x1 = torch.zeros(16, 16, device='cuda', dtype=torch.float8_e4m3fn) 
x2 = torch.zeros(16, 16, device='cuda', dtype=torch.float8_e4m3fn).t() 
res = torch._scaled_mm(x1, x2)
print('done')
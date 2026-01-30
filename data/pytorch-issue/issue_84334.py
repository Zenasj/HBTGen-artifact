import torch
device = 'cuda:0'
torch.backends.cuda.matmul.allow_tf32 = True
dtype = torch.float32
identity = torch.tensor([[1.0]], dtype=dtype, device=device)
vector = torch.tensor([1024.5], dtype=dtype, device=device)

product = vector.matmul(identity)
print(product)

import torch
import numpy as np

torch.backends.cuda.matmul.allow_tf32 = True
device = 'cuda:0'
dtype = torch.float32
a = np.load('trya.npy') # shape = (129, 129)
b = np.load('tryn.npy') # shape = (1, 129, 3)
a = torch.from_numpy(a).to(device).to(dtype)
b = torch.from_numpy(b).to(device).to(dtype)

r1 = torch.matmul(a, b)
r2 = torch.matmul(a.cpu(), b.cpu())
gt = torch.matmul(a.to(torch.float64), b.to(torch.float64))

print('Comparison 1')
print((r1.cpu() - r2).abs().sum())
print((r1.cpu() - gt.cpu()).abs().sum())
print((r2.cpu() - gt.cpu()).abs().sum())

a = torch.zeros(100, 100).uniform_().to(device).to(dtype)
b = torch.zeros(100, 100).uniform_().to(device).to(dtype)

r1 = torch.matmul(a, b)
r2 = torch.matmul(a.cpu(), b.cpu())
gt = torch.matmul(a.to(torch.float64), b.to(torch.float64))


print('Comparison 2')
print((r1.cpu() - r2).abs().sum())
print((r1.cpu() - gt.cpu()).abs().sum())
print((r2.cpu() - gt.cpu()).abs().sum())
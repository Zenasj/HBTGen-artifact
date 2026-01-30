import torch

A = torch.rand([5, 5], dtype=torch.float64, requires_grad=True)
B = torch.rand([5], dtype=torch.float64, requires_grad=True)

res = torch.linalg.lstsq(A, B)
res.solution.sum().backward()
# IndexError: Dimension out of range (expected to be in range of [-1, 0], but got -2)

import torch
print(f'Running PyTorch version: {torch.__version__}')

torchdevice = torch.device('cpu')
if torch.cuda.is_available():
  torchdevice = torch.device('cuda')
  print('Default GPU is ' + torch.cuda.get_device_name(torch.device('cuda')))
print('Running on ' + str(torchdevice))

# Test matrix-vector solver
Aref = torch.randn(5,3, dtype=torch.float64, requires_grad=False, device=torchdevice)
bref = torch.randn(5,1, dtype=torch.float64, requires_grad=False, device=torchdevice)

A1 = Aref.detach().clone().requires_grad_(True)
b1 = bref.detach().clone().requires_grad_(True)

# Solve
x1 = torch.linalg.lstsq(A1,b1).solution
print('x1',x1)

# arbitrary scalar function to mimick a loss
loss1 = x1.sum()
loss1.backward()
print('A1.grad',A1.grad)
print('b1.grad',b1.grad)

# Test matrix-vector solver
A2 = Aref.detach().clone().requires_grad_(True)
b2 = bref.detach().squeeze().clone().requires_grad_(True)

# Solve
x2 = torch.linalg.lstsq(A2,b2).solution
print('x2',x2)

# arbitrary scalar function to mimick a loss
loss2 = x2.sum()
loss2.backward()
print('A2.grad',A2.grad)
print('b2.grad',b2.grad)
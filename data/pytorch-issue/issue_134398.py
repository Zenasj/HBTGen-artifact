import torch

t1 = torch.tensor(5., requires_grad=True) # Leaf tensor
t2 = torch.tensor(2., requires_grad=True) # Leaf tensor

t3 = t1 + t2 # Non-leaf tensor

print(t3.grad) # Warning

t3.backward()

print(t3.grad) # Warning

import torch

t1 = torch.tensor(5., requires_grad=True) # Leaf tensor
t2 = torch.tensor(2., requires_grad=True) # Leaf tensor

t3 = t1 + t2 # Non-leaf tensor

t3.retain_grad() # Here

print(t3.grad) # None

t3.backward()

print(t3.grad) # tensor(1.)

import torch

t1 = torch.tensor(5., requires_grad=True) # Leaf tensor
t2 = torch.tensor(2., requires_grad=True) # Leaf tensor

t3 = t1 + t2 # Non-leaf tensor

t3.grad = torch.tensor(8.) # Here

print(t3.grad) # tensor(8.)

t3.backward()

print(t3.grad) # tensor(8.)

import torch

torch.__version__ # 2.3.1+cu121
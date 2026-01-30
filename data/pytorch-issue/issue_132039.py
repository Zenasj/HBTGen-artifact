import torch
# Create a square matrix
a = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
# Calculate the condition number
cond_num = torch.linalg.cond(a)
print("Condition number (ord=2):", cond_num) # Condition number (ord=2): tensor(1.6336e+08)

import torch
# Create a square matrix
a = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]).cuda()
# Calculate the condition number
cond_num = torch.linalg.cond(a)
print("Condition number (ord=2):", cond_num) # Condition number (ord=2): tensor(2.9279e+08, device='cuda:0')
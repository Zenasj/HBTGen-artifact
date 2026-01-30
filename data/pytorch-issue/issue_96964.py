import torch
A = torch.ones(5, 5).to('cuda')
B = torch.ones(5, 5).to('cuda')
C = torch.matmul(A, B)
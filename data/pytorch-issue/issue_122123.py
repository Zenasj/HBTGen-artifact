import torch

A = torch.randn(32769, 4, device='mps')
B = A[1:25] @ torch.randn(4, 10, device='mps')

A = torch.randn(1, 100000, device="mps")
B = torch.randn(10, 1, device="mps")
A = A[:, 16384:32769]
print(torch.mm(B, A))

import timeit

a_cpu = torch.rand(250, device='cpu')
b_cpu = torch.rand((250, 250), device='cpu')
a_mps = torch.rand(250, device='mps')
b_mps = torch.rand((250, 250), device='mps')

print('cpu', timeit.timeit(lambda: a_cpu @ b_cpu, number=100_000))
print('mps', timeit.timeit(lambda: a_mps @ b_mps, number=100_000))
import torch

a = torch.eye(5)
a[0, :] = 0
b = torch.ones(5)
for _ in range(100):
    sol = torch.linalg.lstsq(a, b)
    x = sol.solution
    print('Rank =', int(sol.rank), 'Error =', float(torch.sum((a @ x - b)**2)))
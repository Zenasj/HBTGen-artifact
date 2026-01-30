import torch

n_rows = 5000
n_cols = 1000000

for i in range(10):
    tmp = torch.zeros((n_rows, n_cols), dtype=torch.float, device=torch.device('cpu'))
    n_cols -= 10000

for i in range(10):
    if i == 0:
        tmp = torch.zeros((n_rows, n_cols), dtype=torch.float, device=torch.device('cpu'))
    else:
        tmp = torch.zeros((n_rows, n_cols), out=tmp, dtype=torch.float, device=torch.device('cpu'))
    n_cols -= 10000
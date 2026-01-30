import torch
import os

cwd = os.getcwd()

z = torch.load(cwd + '/z.pt')
cov = (torch.cov(z.t()))

U, S, VT = torch.linalg.svd(cov)

print(S)
print(S.shape)
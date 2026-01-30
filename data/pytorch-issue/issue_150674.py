import torch
input = torch.randn(2, 3, 4)
tau = torch.randn(2, 2)
other = torch.randn(2, 4, 3)
output = torch.ormqr(input, tau, other, left=False, transpose=False)
print(input)
print(tau)
print(other)
print(output)
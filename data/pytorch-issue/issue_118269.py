import torch

x = torch.randn(1000, 1000, dtype=torch.complex64)
print('Complex signal power using native complex: ', torch.var(x))
print('Real part signal power using native complex: ', torch.var(torch.real(x)))
print('Imag part signal power using native complex: ', torch.var(torch.imag(x)))
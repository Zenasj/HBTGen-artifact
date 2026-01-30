import torch

a = torch.zeros((5, 6, 7), dtype=torch.complex64, device='cpu')
print(a.shape)
print(a.stride())
b = torch._fft_c2c(a, [0], 2, False)
print(b.shape)
print(b.stride())

torch.Size([5, 6, 7])
(42, 7, 1)
torch.Size([5, 6, 7])
(1, 35, 5)
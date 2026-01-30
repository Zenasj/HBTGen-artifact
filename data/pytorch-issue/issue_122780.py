import torch

x = torch.randn(1, 16000).float()
y1 = torch.stft(x, n_fft=256, hop_length=160, return_complex=False)
y2 = torch.stft(x, n_fft=256, hop_length=160, return_complex=True)
y3 = torch.view_as_real(y2)

y1 = torch.sqrt(torch.sum(y1 ** 2, dim=-1))
y2 = y2.abs()
y3 = torch.sqrt(torch.sum(y3 ** 2, dim=-1))

print(torch.max(torch.abs(y2 - y1)))
print(torch.max(torch.abs(y3 - y1)))

import torch

x = torch.randn(1, 16000).float()
y1 = torch.stft(x, n_fft=256, hop_length=160, return_complex=False)
y2 = torch.stft(x, n_fft=256, hop_length=160, return_complex=True)
y3 = torch.view_as_real(y2)
print(torch.max(torch.abs(y3 - y1)))

import torch

x = torch.randn(1, 16000).float()
y2 = torch.stft(x, n_fft=256, hop_length=160, return_complex=True)
print(torch.max(torch.abs(y2.abs() - torch.sqrt(torch.sum(torch.view_as_real(y2) ** 2, dim=-1)))))
# tensor(3.8147e-06)

import torch

a = torch.randn(100, dtype=torch.complex64)
b1 = a.abs()
b2 = torch.hypot(a.real, a.imag)
(b1 - b2).abs().max()
tensor(0.)

import torch

a = torch.randn(100, dtype=torch.complex64)
a_64 = a.to(torch.complex128)
expect = torch.sqrt(a_64.real.square() + a_64.imag.square())
b1 = a.abs()
b2 = torch.sqrt(a.real.square() + a.imag.square())

print((b1 - expect).abs().max()) # tensor(5.8497e-08, dtype=torch.float64)
print((b2 - expect).abs().max()) # tensor(1.1215e-07, dtype=torch.float64)
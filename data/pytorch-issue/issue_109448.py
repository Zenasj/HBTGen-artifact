import torch

device = 'cuda'

x = torch.randn(32, dtype=torch.cfloat, device=device, requires_grad=True)
dout = torch.zeros(32, dtype=torch.cfloat, device=device)

# compute iFFT(FFT(x))
out = torch.fft.ifft(torch.fft.fft(x))
out.backward(dout, retain_graph=True)

print('Gradient of iFFT(FFT(x)) should be FFT(iFFT(dout))')
dx = torch.fft.fft(torch.fft.ifft(dout))

print('Difference between x.grad and what it should be. This should be zero!')
print((x.grad - dx).abs().max())

print('Difference between x.grad and x. This should be non-zero.')
print((x.grad - x).abs().max())
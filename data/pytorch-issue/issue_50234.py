import torch
a = torch.rand(100, dtype=torch.complex64)
torch.stft(a, n_fft=10)
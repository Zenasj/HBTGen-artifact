import torch

S_torch = torch.fft(x_torch.transpose(1, -2), signal_ndim=1).transpose(1, -2)
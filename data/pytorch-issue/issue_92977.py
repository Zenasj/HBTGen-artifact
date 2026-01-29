# torch.rand(B, T, dtype=torch.float32)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def pad_center(data, *, size, axis=-1, **kwargs):
    kwargs.setdefault("mode", "constant")
    n = data.shape[axis]
    lpad = int((size - n) // 2)
    lengths = [(0, 0)] * data.dim()
    lengths[axis] = (lpad, int(size - n - lpad))
    if lpad < 0:
        raise ValueError(f"Target size ({size:d}) must be at least input size ({n:d})")
    pad_tuple = ()
    for l in lengths:
        for li in l:
            pad_tuple += (li,)
    return F.pad(data, pad_tuple, **kwargs)

class STFT(torch.nn.Module):
    def __init__(self, win_length, hop_length, nfft, window, window_periodic=True, stft_mode=None):
        super().__init__()
        self.win_length = win_length
        self.hop_length = hop_length
        self.nfft = nfft
        self.freq_cutoff = self.nfft // 2 + 1
        self.window = window

        if stft_mode == 'conv':
            fourier_basis = torch.view_as_real(torch.fft.fft2(torch.eye(self.nfft), dim=1))
            forward_basis = fourier_basis[:self.freq_cutoff].permute(2, 0, 1).reshape(-1, 1, fourier_basis.shape[1])
            forward_basis = forward_basis * pad_center(self.window, size=self.nfft)
            self.stft = torch.nn.Conv1d(
                forward_basis.shape[1],
                forward_basis.shape[0],
                forward_basis.shape[2],
                bias=False,
                stride=self.hop_length
            ).requires_grad_(False)
            self.stft.weight.copy_(forward_basis)
        else:
            self.stft = None

    def forward(self, signal):
        pad = self.freq_cutoff - 1
        padded_signal = F.pad(signal.unsqueeze(1), (pad, pad), mode='reflect').squeeze(1)
        if self.stft is not None:
            real_imag = self.stft(padded_signal.unsqueeze(1))
            real, imag = real_imag.split(self.freq_cutoff, dim=1)
        else:
            real, imag = padded_signal.stft(self.nfft, hop_length=self.hop_length, win_length=self.win_length,
                                            window=self.window, center=False).unbind(dim=-1)
        realimag = torch.stack((real, imag), dim=3)
        return realimag

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.window = torch.hamming_window(30)
        self.stft = STFT(30, 160, 1024, self.window, window_periodic=True, stft_mode='conv')
        self.max_dim = 2
        self.bn = nn.BatchNorm2d(1, affine=False)

    def forward(self, x):
        x = x.squeeze(-1)
        x = self.stft(x)
        x, _ = x.max(self.max_dim)
        x = x.unsqueeze(1)
        score = self.bn(x)
        return score

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 60000, dtype=torch.float32)


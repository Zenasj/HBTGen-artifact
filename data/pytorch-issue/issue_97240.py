center

import torch
a = torch.rand(100)
b1 = torch.stft(a, n_fft=100, hop_length=25, win_length=5, center=False, return_complex=True, onesided=False)
b2 = torch.stft(a, n_fft=100, hop_length=25, win_length=5, center=True, return_complex=True, onesided=False)

n_frames = (input_length - n_fft) / hop_length + 1
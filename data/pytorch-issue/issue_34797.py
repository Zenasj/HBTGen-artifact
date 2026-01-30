import torch

py
batch=5
time=100
n_fft=16
hop=8
input=torch.rand((batch, time, 2))
torch.stft(input, n_fft=n_nfft, hop_length=hop)
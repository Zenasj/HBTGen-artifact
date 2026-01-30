import torchaudio

waveform, sample_rate = torchaudio.load('test.wav')
spectrogram = torchaudio.transforms.Spectrogram(sample_rate)(waveform)

import torchaudio
import torch

waveform, sample_rate = torchaudio.load('test.wav')
waveform = waveform.to("cuda:0")

spectrogram = torchaudio.transforms.Spectrogram(sample_rate).to("cuda:0")(waveform)

import torch
x = torch.randn(10,10,2)
torch.fft(x,1)
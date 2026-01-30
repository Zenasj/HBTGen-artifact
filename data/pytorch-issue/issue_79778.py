import random

import numpy as np

import torch 
import torchaudio
import librosa

print(f"torch version: {torch.__version__}")
print(f"librosa version: {librosa.__version__}")

center = False

### librosa
my_audio = np.random.randn(25600)
my_stft  = librosa.stft(my_audio, win_length=1024, hop_length=256, n_fft=1024, center=center)
my_reconstructued_audio = librosa.istft(my_stft, hop_length=256, center=center)
print("-----------------")
print("librosa:")
print(my_audio.shape)
print(my_reconstructued_audio.shape)

### pytorch
my_audio = torch.randn(1, 25600)
my_stft  = torch.stft(my_audio, win_length=1024, hop_length=256, n_fft=1024, center=center, return_complex=True)
my_reconstructued_audio = torch.istft(my_stft, n_fft=1024, hop_length=256, center=center)
print("-----------------")
print("pytorch:")
print(my_audio.shape)
print(my_reconstructued_audio.shape)
import torch
import librosa

print(torch.__version__) # 1.3.1
print(librosa.__version__) # 0.6.3

x, sr = librosa.load(librosa.util.example_audio_file(), offset=15.0, duration=5.0)
print(x.shape) # (110250,)
x = torch.from_numpy(x).cuda()
window = torch.hann_window(window_length=400)

torch.stft(x, n_fft=512, hop_length=160, win_length=400, window=window).shape # torch.Size([257, 690, 2])

torch.stft(x, n_fft=400, hop_length=160, win_length=400, window=window).shape # RuntimeError: expected device cuda:0 but got device cpu
import torch

fname = 'stft.pt'
stft_feat = torch.load(fname)
print("stft_feat", stft_feat.size())

n_fft = 512
hop_length = 320
win_length = 512
window = torch.hann_window(win_length).cuda()
center = True
length = 150079
reconstruct = torch.istft(stft_feat, n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, length=length)[0]
print("reconstruct", reconstruct.size())
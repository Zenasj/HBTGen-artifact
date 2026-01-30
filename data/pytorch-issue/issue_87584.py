channels_out = (in_channels-(kernel_size-1)-1)/stride+1

import torch

in_channels = 1000
stride = 100
kernel_size = 16
input = torch.zeros((in_channels,))

for i in range(4, 9):
    n_fft = 2**i
    output = torch.stft(input, n_fft=n_fft, hop_length=stride, win_length=kernel_size, center=False, return_complex=True)
    expected_channels_out = (in_channels-(kernel_size-1)-1)/stride+1
    actual_channels_out = (in_channels-(n_fft-1)-1)/stride+1
    print(f"{output.shape=}, {expected_channels_out=}, {actual_channels_out=}")
import torch

x = torch.rand(batchsize, channels, 10,10)
x = pad(x, padding=(10, 10, 10, 10), padding_mode='circular')
x = pad(x, padding=(5, 5, 5, 5), padding_mode='circular')
conv = Conv2d(10, 10, kernal_size=(3,3), padding=0, dilation=15)
x = conv(x)

x = torch.rand(batchsize, channels, 10,10)
conv = Conv2d(10, 10, kernel_size=(3,3), padding=15, dilation=15, padding_mode = 'circular')
x = conv(x)
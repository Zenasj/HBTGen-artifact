import torch.nn as nn

import torch
class Stft(torch.nn.Module):
    def forward(self, x):
        y = torch.stft(x, n_fft=512)
        return y
a = Stft()
x = torch.randn(2, 2000)
torch.nn.parallel.data_parallel(a, (x,), device_ids=[0])
print("single gpu passed")
torch.nn.parallel.data_parallel(a, (x,), device_ids=[0, 1])
print("two gpu passed")
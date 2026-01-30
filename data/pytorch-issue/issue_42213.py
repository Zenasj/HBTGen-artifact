import time

import torch
import torchaudio

n_fft = 2**15
x = torch.rand(size=[1, n_fft//2 + 1, 10, 2])

start = time.monotonic()
n = 10
for _ in range(n):
    # y = torchaudio.functional.istft(x,n_fft)
    y = torch.functional.istft(x,n_fft)
elapsed = time.monotonic() - start

print(elapsed, elapsed / n)
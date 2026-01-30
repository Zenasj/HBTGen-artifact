import torch
import numpy as np

SIZE=19

print(torch.all(torch.isfinite(torch.fft.fft(torch.eye(SIZE), dim=1))))
np.exp([2])
print(torch.all(torch.isfinite(torch.fft.fft(torch.eye(SIZE), dim=1))))
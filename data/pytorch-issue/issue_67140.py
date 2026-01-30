import torch

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from torch.fft import rfft, irfft
device = 'cuda'
pdf = torch.from_numpy(
    norm.pdf(np.linspace(-10, 10, 1000).reshape(-1,1))
).to(device)
n_iter = 50
n_full = pdf.shape[0] * n_iter - n_iter + 1
fft0 = rfft(pdf, axis=0, n=n_full)
fft1 = rfft(pdf, axis=0, n=n_full)
plt.plot(np.abs(fft0.cpu()[:, 0]), label='fft0')
plt.plot(np.abs(fft1.cpu()[:, 0]), label='fft1')
plt.legend()
plt.title(f'On {device}')
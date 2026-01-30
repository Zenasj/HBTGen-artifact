import torch
import matplotlib.pyplot as plt

T, B = 8, 16

ts, bs = torch.meshgrid(
        torch.arange(T, device='cuda'),
        torch.arange(B, device='cuda'))

ts_inv = ts.flip(0)
ts[ts_inv, bs] = ts

plt.imshow(ts.cpu())

import numpy as np

T, B = 8, 16

ts, bs = np.meshgrid(
        np.arange(T),
        np.arange(B), indexing='ij')

ts_inv = ts[::-1]
ts[ts_inv, bs] = ts

plt.imshow(ts)
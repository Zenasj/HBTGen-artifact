import torch.nn as nn

import torch
import torch.fft as fft
import numpy as np
from scipy.fft import rfft

func_cls=torch.fft.rfftn
# func_cls=torch.fft.rfft2


if __name__ == '__main__':
    dtype = torch.float64
    for device in ['cuda', 'cpu']:
        print('\ndevice={}'.format(device))
        Ly = 1600
        y = torch.randn(1, 1, Ly, dtype=dtype, device=device)
        for Lx in [16000, 16001]:
            x = torch.randn(1, 1, Lx, dtype=dtype, device=device)
            x = torch.nn.functional.pad(x, [0, Ly-1])
            yy = torch.nn.functional.pad(y, [0, Lx-1])
            X = func_cls(x)
            Y = func_cls(yy)
            X_s = rfft(x[0, 0, :].cpu().numpy())
            Y_s = rfft(yy[0, 0, :].cpu().numpy())
            dif_x = np.max(np.abs(X_s - X[0, 0, :].cpu().numpy()))
            dif_y = np.max(np.abs(Y_s - Y[0, 0, :].cpu().numpy()))
            print('Lx={}: dif_x={}, dif_y={}'.format(Lx, dif_x, dif_y))

import numpy as np
import torch
from scipy.stats import norm
from torch.fft import rfft
torch.manual_seed(0)
# func_cls=torch.fft.rfft
# func_cls=torch.fft.ifft
func_cls=torch.fft.hfft

device = 'cuda'
# device = 'cpu'
pdf = torch.from_numpy(
    norm.pdf(np.linspace(-10, 10, 1000).reshape(-1,1))
).to(device)
n_iter = 50
n_full = pdf.shape[0] * n_iter - n_iter + 1
fft0 = func_cls(pdf, axis=0, n=n_full)
fft1 = func_cls(pdf, axis=0, n=n_full)
x1=np.abs(fft0.cpu()[:, 0])
x2=np.abs(fft1.cpu()[:, 0])
print(x1)
print(x2)
print(np.allclose(x1,x2))
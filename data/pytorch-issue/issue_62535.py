import torch.nn as nn
import random

import torch
from torch import nn, optim
import numpy as np
from utils import fft2c_new, ifft2c_new, complex_abs

device = 'cuda:0'

model = nn.Conv2d(1, 1, kernel_size=(3, 3), padding=1).to(device)
Niter = 10000
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss(reduction='mean')

# Helper functions

def kspace_to_nchw(tensor):
    """
    Convert torch tensor in (Slice, Coil, Height, Width, Complex) 5D format to
    (N, C, H, W) 4D format for processing by 2D CNNs.
    """
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dim() == 5
    s = tensor.shape
    assert s[-1] == 2
    tensor = tensor.permute(dims=(0, 1, 4, 2, 3)).reshape(shape=(s[0], 2 * s[1], s[2], s[3]))
    return tensor


def nchw_to_kspace(tensor):
    """
    Convert a torch tensor in (N, C, H, W) format to the (Slice, Coil, Height, Width, Complex) format.
    """
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dim() == 4
    s = tensor.shape
    if tensor.shape[1] == 1:
        imag_tensor = torch.zeros(s, device=tensor.device)
        tensor = torch.cat((tensor, imag_tensor), dim=1)
        s = tensor.shape
    tensor = tensor.view(size=(s[0], s[1] // 2, 2, s[2], s[3])).permute(dims=(0, 1, 3, 4, 2))
    return tensor


def get_mask(img, size, acc_factor=8, device='cuda:0'):
    mux_in = size ** 2
    Nsamp = mux_in // acc_factor
    mask = torch.zeros_like(img, dtype=torch.float32, device=device)
    cov_factor = size * (1.5 / 128)
    mean = [size // 2, size // 2]
    cov = [[size * cov_factor, 0], [0, size * cov_factor]]
    samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
    int_samples = samples.astype(int)
    mask[..., int_samples[:, 0], int_samples[:, 1]] = 1
    return mask

# Main script

for step, i in enumerate(range(Niter), start=1):
    optimizer.zero_grad()
    img = torch.randn([1, 1, 128, 128], device=device)
    target = torch.randn([1, 1, 128, 128], device=device)

    kspace = kspace_to_nchw(fft2c_new(nchw_to_kspace(img)))
    mask = get_mask(kspace, 128,
                    acc_factor=8,
                    device=device)
    # this is where the error stems from
    # u_kspace = torch.where(mask > 0., kspace, torch.tensor(0., dtype=torch.float32, device=device))
    u_kspace = kspace * mask
    img_complex = ifft2c_new(nchw_to_kspace(u_kspace))
    img_abs = complex_abs(img_complex)
    est = model(img_abs)
    loss = loss_fn(est, target)
    loss.backward()
    optimizer.step()
    # log
    print(f'step: {step}/{Niter}')
    print(f'loss: {loss.data}')

samples = np.random.multivariate_normal(mean, cov, int(Nsamp))
int_samples = samples.astype(int)
mask[..., int_samples[:, 0], int_samples[:, 1]] = 1

int_samples = np.clip(int_samples, 0, size - 1)
import torch
import numpy as np
from sklearn.gaussian_process.kernels import Matern

## Cholesky decomposition for large positive definite matrix (from GP matern kernel) ##

def matern_kernel_cov(grids, length_scale, nu):
    """
    grids : [n_points, 1 or 2]
    """
    kernel = 1.0 * Matern(length_scale=length_scale, length_scale_bounds="fixed", nu=nu)
    return kernel(grids)

def make_grid(dims, x_min=0, x_max=1):
    # Makes a 2D grid in the format of (n_grid, 2)
    x1 = torch.linspace(x_min, x_max, dims[0])
    x2 = torch.linspace(x_min, x_max, dims[1])
    x1, x2 = torch.meshgrid(x1, x2, indexing='ij')
    grid = torch.cat((
        x1.contiguous().view(x1.numel(), 1),
        x2.contiguous().view(x2.numel(), 1)),
        dim=1)
    return grid

## Matern kernel for Gaussian Random Field
dims = [64,64]
length_scale_x = 0.4
nu_x = 1.5

device = 'cuda:0'
jitter = 1e-6
## kernel shape: [N, N], mean shape :[N]
# dims should be 1D [n_x] or 2D [n_x, n_x]
n_points = np.prod(dims)
grids = make_grid(dims)
matern_ker = matern_kernel_cov(grids, length_scale_x, nu_x)

base_mu = torch.zeros(n_points).float()
#add jitter 
base_cov = torch.tensor(matern_ker).float() #+ jitter * torch.eye(matern_ker.shape[0])   

################################### Comparison, calculated on GPU and CPU #########################
torch.linalg.cholesky_ex(base_cov.to(device))
## output: torch.return_types.linalg_cholesky_ex(
""" # numerical instability, considering jitter doesn't help
L=tensor([[1.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.9977, 0.0671, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.9914, 0.1282, 0.0273,  ..., 0.0000, 0.0000, 0.0000],
        ...,
        [0.0169, 0.0111, 0.0058,  ...,    nan, 0.0000, 0.0000],
        [0.0163, 0.0107, 0.0056,  ...,    nan,    nan, 0.0000],
        [0.0156, 0.0104, 0.0055,  ...,    nan,    nan,    nan]],
       device='cuda:0'),
info=tensor(2347, device='cuda:0', dtype=torch.int32))
"""

torch.linalg.cholesky_ex(base_cov)
""" # it's correct on CPU
torch.return_types.linalg_cholesky_ex(
L=tensor([[1.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.9977, 0.0671, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.9914, 0.1282, 0.0273,  ..., 0.0000, 0.0000, 0.0000],
        ...,
        [0.0169, 0.0111, 0.0058,  ..., 0.0127, 0.0000, 0.0000],
        [0.0163, 0.0107, 0.0056,  ..., 0.0132, 0.0129, 0.0000],
        [0.0156, 0.0104, 0.0055,  ..., 0.0097, 0.0146, 0.0150]]),
info=tensor(0, dtype=torch.int32))
"""
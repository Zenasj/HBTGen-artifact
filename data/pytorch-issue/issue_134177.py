import torch

from torch import einsum, ones
n_samples = 17
einsum('b i d, b j d -> b i j', ones(16 * n_samples, 4096, 40, device='mps'), ones(16 * n_samples, 4096, 40, device='mps')).shape
# Output:
# RuntimeError: Tiling of batch matmul for larger than 2**32 entries only available from MacOS15 onwards

from torch import einsum, ones
n_samples = 17
einsum('b i d, b j d -> b i j', ones(16 * n_samples, 4096, 40, device='mps'), ones(16 * n_samples, 4096, 40, device='mps')).shape
# torch.Size([272, 4096, 4096])
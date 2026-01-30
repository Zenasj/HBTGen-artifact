import torch
import numpy as np

def cholesky(x, *args, **kwargs):
    shape = x.shape
    batch_size = np.prod(shape[:-2])
    cuda_batch_size = (1 << 19) - 8

    if batch_size <= cuda_batch_size:
        return torch.cholesky(x, *args, **kwargs)

    x = x.view(-1, shape[-1], shape[-1])
    acc = torch.empty_like(x)
    for i in range(0, batch_size, cuda_batch_size):
        indices = torch.arange(
                i,
                min(batch_size, i + cuda_batch_size),
                device=x.device,
                dtype=torch.int64)
        acc[indices] = torch.cholesky(x[indices], *args, **kwargs)

    return acc.view(shape)

torch.cholesky(torch.eye(2, device='cuda').expand((1 << 19) - 7, -1, -1))
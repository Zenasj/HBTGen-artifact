import torch

from tenacity import retry, stop_after_attempt


@retry(stop=stop_after_attempt(32)) # Or some other value
def func_with_svd(L: torch.Tensor):
    try:
        u, s, v = torch.svd(L)
    except:                     # torch.svd may have convergence issues for GPU and CPU.
        u, s, v = torch.svd(L + 1e-4*L.mean()*torch.rand_like(L))
    ...
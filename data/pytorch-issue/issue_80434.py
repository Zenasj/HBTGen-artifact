from  torch.testing._internal.common_utils import freeze_rng_state
import torch

def fn(x):
    torch.manual_seed(42)
    return torch.rrelu(x, training=True)

def fn2(x):
    with freeze_rng_state():
        return torch.rrelu(x, training=True)

for device in ("cpu", "cuda"):  # Fails when CUDA
    a = torch.rand(10, 10, 10, requires_grad=True, dtype=torch.float64, device=device)

    for fast_mode in (True, False):  # Doesn't affect results
        print(fast_mode, device)
        print(torch.autograd.gradcheck(fn, (a,), fast_mode=fast_mode, raise_exception=False))
        print(torch.autograd.gradcheck(fn2, (a,), fast_mode=fast_mode, raise_exception=False))
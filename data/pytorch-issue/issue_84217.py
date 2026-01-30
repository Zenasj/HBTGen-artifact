Python
import warnings
import torch
import random
import numpy as np
import torch.utils.benchmark as benchmark

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    device='cuda'
    dtype=torch.float16

    a = torch.randn(512, 1024, device=device, dtype=dtype, requires_grad=False)
    b = torch.randn(1024, 1024, device=device, dtype=dtype, requires_grad=True)
    x = torch.nested_tensor([a, b])
    nt_size = x._nested_tensor_size()

    x = x.to_padded_tensor(0)
    x = x.transpose(-1,-2)
    # x = x.contiguous()

    torch._nested_from_padded(x, nt_size)
    t0 = benchmark.Timer(
    stmt='torch._nested_from_padded(x, nt_size)',
    globals={'x': x, 'nt_size': nt_size},
    label=f'Testing generic kernel with contiguous calls',
    num_threads=torch.get_num_threads())
    m0 = t0.blocked_autorange(min_run_time=10)
    print(m0)
import torch
import numpy as np
import pytest

torch.set_num_threads(1)
torch.set_grad_enabled(False)

x = torch.randn(1000, 1000, dtype=torch.float64)
q = torch.rand(1, dtype=torch.float64)


@pytest.mark.parametrize('op', ['torch', 'numpy'])
def test_quantile(benchmark, op):
    if op == 'torch':
        benchmark(torch.nanquantile, x, q, 1)
    else:
        benchmark(np.nanquantile, x.numpy(), q.numpy(), 1)
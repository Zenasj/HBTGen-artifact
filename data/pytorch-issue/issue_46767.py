import torch

torch.set_num_threads(1)
x = torch.randn(1000, 10, 36, requires_grad=True)

def test_grad_enabled(benchmark):
    benchmark(torch.max_pool1d, x, 2)

def test_grad_disabled(benchmark):
    torch.set_grad_enabled(False)
    benchmark(torch.max_pool1d, x, 2)
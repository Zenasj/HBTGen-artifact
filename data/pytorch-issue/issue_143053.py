import torch

def sum_combine(a, b):
    return a + b

from torch._higher_order_ops.associative_scan import associative_scan

a = torch.randn(100, 100, device=torch.device("cuda"))
expect = torch.cumsum(a, 0)
actual = associative_scan(sum_combine, a, 0)

def logcumsum_combine(a, b):
    min_v = torch.minimum(a, b)
    max_v = torch.maximum(a, b)
    mask = (min_v != max_v) | ~min_v.isinf()
    return torch.where(mask, max_v + (min_v - max_v).exp().log1p(), a)

expect = torch.logcumsumexp(a, 0)
actual = associative_scan(logcumsum_combine, a, 0)
print(expect, actual)
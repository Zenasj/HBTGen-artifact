import torch.nn as nn

Python
import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

q = torch.randn(64, 1024, 8, 64, dtype=torch.half, device='cuda')
print(torch._C._get_sdp_priority_order())

orders = [[SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION],
          [SDPBackend.MATH, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION],
          [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]]
import time
times = list()
for order in orders:
    print(order)
    with sdpa_kernel(order, set_priority=True):
        scaled_dot_product_attention(q, q, q)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with sdpa_kernel(order, set_priority=True):
        scaled_dot_product_attention(q, q, q)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append(t1 - t0)
print(times)
assert times[0] < times[1]
assert times[0] > times[2]
assert times[1] > times[2]
print(torch._C._get_sdp_priority_order())
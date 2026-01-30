import torch.nn as nn

import torch
import time

torch.set_default_device("cuda")
torch.set_default_dtype(torch.float16)

SHAPE = (8, 64, 1024, 1024)

@torch.inference_mode()
def benchmark(n: int) -> float:
    now = time.perf_counter()
    for _ in range(n):
        torch.nn.functional.scaled_dot_product_attention(torch.randn(SHAPE), torch.randn(SHAPE), torch.randn(SHAPE))
    return time.perf_counter() - now

benchmark(100)  # warmup
print(f"100 steps of SDPA completed in {benchmark(100) * 1000}ms")
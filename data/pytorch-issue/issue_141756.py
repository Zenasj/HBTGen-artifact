import torch.nn as nn

import time
import torch
from torch import Tensor
from torch.nn.attention import sdpa_kernel, SDPBackend
import torch.nn.functional as F


def get_device_name():
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    return device_name


def sdpa(q: Tensor, k: Tensor, v: Tensor):
    with sdpa_kernel(SDPBackend.MATH):
        out: Tensor = F.scaled_dot_product_attention(q, k, v)
    return out


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    q = torch.randn((1, 50176, 256), device="cuda", dtype=torch.half)
    k = torch.randn((1, 50176, 256), device="cuda", dtype=torch.half)
    v = torch.randn((1, 50176, 256), device="cuda", dtype=torch.half)
    torch.cuda.synchronize()

    repeat = 100
    warmup = 5
    
    # warmup
    for _ in range(warmup):
        with torch.autocast(device_type="cuda", 
                            dtype=torch.float16):
            out = sdpa(q, k, v)
    torch.cuda.synchronize()

    # repeat
    start = time.time()
    for _ in range(repeat):
        with torch.autocast(device_type="cuda", 
                            dtype=torch.float16):
            out = sdpa(q, k, v)
            
    torch.cuda.synchronize()
    end = time.time()
    time_cost_ms = (end - start) * 1000 / repeat
    free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
    total_use_memory = (total_gpu_memory - free_gpu_memory) / (1024 ** 3)
    print(f"device: {get_device_name()}, torch version: {torch.__version__}, repeat: {repeat}, "
          f"mean time: {time_cost_ms}ms, memory usage:{total_use_memory}GiB")

q = torch.randn((1, 65536, 128), device="cuda", dtype=torch.half)
k = torch.randn((1, 65536, 128), device="cuda", dtype=torch.half)
v = torch.randn((1, 65536, 128), device="cuda", dtype=torch.half)
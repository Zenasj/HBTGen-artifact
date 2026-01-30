import gc
import os
import torch
import psutil


def memory_usage_psutil():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"use mem {mem_info.rss / 1024 / 1024 /4}MB")  # rss即Resident Set Size，表示进程占用的物理内存


@torch.no_grad()
def _get_kth_value(value: torch.Tensor, kth_idx: int):
    _k = torch.kthvalue(value, k=kth_idx, dim=0)[0]
    _k = value[0]
    kth_val = _k
    return kth_val


@torch.no_grad()
def percentile(value):

    numel = value.numel()
    _percentile = 0.99999
    min_idx, max_idx = int(numel * (1 - _percentile)), int(numel * _percentile)
    min_idx = 1
    max_idx = 1
    # Clone the flattened tensor to avoid referencing the original data
    flat_value = value
    _min = _get_kth_value(flat_value, min_idx)
    _max = _get_kth_value(flat_value, max_idx)

    return (_max, _min)


_c = []

for i in range(100):
    data = torch.randn(1, 3, 640, 640)
    _c.append(percentile(data))
    memory_usage_psutil()

# a = torch.tensor([1]).reshape([1,-1])
# b = torch.tensor([1]).reshape([1,-1])

# c = torch.cat([a,b],dim=-1)
# print(c.shape)

import gc
import os
import torch
import psutil


def memory_usage_psutil():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"use mem {mem_info.rss / 1024 / 1024 /4}MB")  # rss即Resident Set Size，表示进程占用的物理内存


@torch.no_grad()
def _get_kth_value(value: torch.Tensor, kth_idx: int):
    _k = torch.kthvalue(value, k=kth_idx, dim=0)[0]
    _k = _k.clone().detach().cpu().item()
    kth_val = _k
    return kth_val


@torch.no_grad()
def percentile(value):

    numel = value.numel()
    _percentile = 0.99999
    min_idx, max_idx = int(numel * (1 - _percentile)), int(numel * _percentile)
    min_idx = 1
    max_idx = 1
    flat_value = value.clone().flatten()
    _min = _get_kth_value(flat_value, min_idx)
    _max = _get_kth_value(flat_value, max_idx)
    gc.collect()
    return (_max, _min)


_c = []

for i in range(100):
    data = torch.randn(1, 3, 640, 640)
    _c.append(percentile(data))
    memory_usage_psutil()

# a = torch.tensor([1]).reshape([1,-1])
# b = torch.tensor([1]).reshape([1,-1])

# c = torch.cat([a,b],dim=-1)
# print(c.shape)
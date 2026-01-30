import torch.nn as nn

from __future__ import annotations

import gc
import time
import pynvml
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


INIT_MEMORY_USED = None
NDIM = 1024 * 1024 * 1024 // 4  # 1GB (4 bytes per element)


def get_memory_stats():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    memory_used, total_memory = 0, 0
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_used += memory_info.used
        total_memory += memory_info.total
    return memory_used / 1024 ** 2, total_memory / 1024 ** 2


def print_memory_used(prefix: str | None = None):
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    global INIT_MEMORY_USED
    torch.cuda.synchronize()
    prefix = prefix or "Total memory used"
    memory_used, total_memory = get_memory_stats()
    if INIT_MEMORY_USED is None:    
        INIT_MEMORY_USED = memory_used
    print(
        f" {prefix}: \033[93m{memory_used - INIT_MEMORY_USED} MB\033[0m"
        f" / \033[92m{total_memory} MB\033[0m"
    )


class MemoryTest(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(NDIM, 1, bias=False)
        
    def forward(self, x):
        return self.layer(x)
    

def offload_model(model: torch.nn.Module):
    for _, param in model.named_parameters():
        if hasattr(param, "_local_shard"):
            param._local_shard = param._local_shard.to("cpu", non_blocking=True)
        param.data = param.data.to("cpu", non_blocking=True)
        if param.grad is not None:
            param.grad = param.grad.to("cpu", non_blocking=True)
    torch.cuda.empty_cache()
            

def reload_model(model: torch.nn.Module):
    for _, param in model.named_parameters():
        if hasattr(param, "_local_shard"):
            param._local_shard = param._local_shard.to("cuda", non_blocking=True)
        param.data = param.data.to("cuda", non_blocking=True)
        if param.grad is not None:
            param.grad = param.grad.to("cuda", non_blocking=True)
    torch.cuda.empty_cache()


def offload_optimizer(optimizer: torch.optim.Optimizer):
    optimizer.zero_grad()
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            state = optimizer.state[param]
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    value.data = value.data.to("cpu", non_blocking=True)
    torch.cuda.empty_cache()


def reload_optimizer(optimizer: torch.optim.Optimizer):
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            state = optimizer.state[param]
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    value.data = value.data.to("cuda", non_blocking=True)
    torch.cuda.empty_cache()
    

def backward(model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    x = torch.randn(1, NDIM).cuda()
    y = model(x)
    y.backward()
    optimizer.step()
    del x, y
    torch.cuda.empty_cache()
    

def main():
    print_memory_used("Initial")
    model = MemoryTest().cuda()
    print_memory_used("After allocating model")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    backward(model, optimizer)
    print_memory_used("After allocating optimizer and back pass")
    offload_model(model)
    print_memory_used("After offloading model")
    offload_optimizer(optimizer)
    print_memory_used("After offloading optimizer")

def fsdp_main():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())

    print_memory_used("Initial")

    model = FSDP(MemoryTest().cuda())
    print_memory_used("After allocating model")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    backward(model, optimizer)
    print_memory_used("After allocating optimizer and back pass")

    offload_model(model)
    print_memory_used("After offloading model")

    offload_optimizer(optimizer)
    print_memory_used("After offloading optimizer")

    dist.destroy_process_group()


if __name__ == "__main__":
    fsdp_main()

def backward(model: torch.nn.Module):
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, NDIM).cuda()
        y = fsdp_forward(model, x)
    gc.collect()
    torch.cuda.empty_cache()
    
def fsdp_forward(model: FSDP, *args, **kwargs):
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        _root_pre_forward, _pre_forward,
        _pre_forward_unshard, _post_forward,
        _post_forward_reshard
    )
    handle = model._handle
    print_memory_used("Before root pre forward")
    args, kwargs = _root_pre_forward(model, model, args, kwargs)
    print_memory_used("After root pre forward")
    unused = None
    args, kwargs = _pre_forward(
        model,
        handle,
        _pre_forward_unshard,
        model._fsdp_wrapped_module,
        args,
        kwargs,
    )
    print_memory_used("After pre forward")
    output = model._fsdp_wrapped_module(*args, **kwargs)
    ans = _post_forward(
        model, handle, _post_forward_reshard, model, unused, output
    )
    print_memory_used("After post forward")
    return ans
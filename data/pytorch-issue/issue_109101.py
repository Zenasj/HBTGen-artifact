# Correct, Poor Performance
model = ...
model = distribute_module(model, mesh, _data_parallel_fn, input_fn=None, output_fn=None)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, amsgrad=False)

# Wrong, Good Performance
model = ...
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, amsgrad=False)
model = distribute_module(model, mesh, _data_parallel_fn, input_fn=None, output_fn=None)

import torch
import torch.nn as nn

import torch.distributed as dist
from torch.distributed._tensor import (
  DTensor,
  DeviceMesh,
  distribute_tensor,
  distribute_module,
  Shard,
  Replicate
)
import torch.multiprocessing as mp
import os
import time

WORLD_SIZE = 1
ITER_TIME = 20

def _data_parallel_fn(
    name: str,
    module: nn.Module,
    device_mesh: DeviceMesh,
) -> None:
    for name, param in module.named_parameters():
        dist_spec = ([Replicate()])
        dist_param = torch.nn.Parameter(
            distribute_tensor(param, device_mesh, dist_spec)
        )
        name = '_'.join(name.split('.'))
        module.register_parameter(name, dist_param)

def native_tensor_baseline():
    in_shape = [20, 32]
    output_shape = [20, 5]

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(1.0, device)

    model = torch.nn.Linear(32, 5).to(device)
    nn.init.ones_(model.weight)
    nn.init.zeros_(model.bias)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, amsgrad=False)

    x = torch.randn(*in_shape).to(device).requires_grad_()
    y_grad = torch.randn(*output_shape).to(device)

    # warm up
    y = model(x)
    optimizer.zero_grad()
    y.backward(y_grad)
    optimizer.step()
    torch.cuda.synchronize(device)

    start = time.time()
    for i in range(ITER_TIME):
        print(f"---------------{i}--------------------")
        torch.cuda.nvtx.range_push("model_iter"+str(i))

        torch.cuda.nvtx.range_push("forward")
        y = model(x)
        torch.cuda.synchronize(device)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("zero_grad")
        optimizer.zero_grad()
        torch.cuda.synchronize(device)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("backward")
        y.backward(y_grad)
        torch.cuda.synchronize(device)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("optimizer_step")
        optimizer.step()
        torch.cuda.synchronize(device)
        torch.cuda.nvtx.range_pop()

        torch.cuda.synchronize(device)
        torch.cuda.nvtx.range_pop()
    end = time.time()
    max_reserved_memory = torch.cuda.max_memory_reserved(device)
    max_allocated_memory = torch.cuda.max_memory_allocated(device)

    print(f"{ITER_TIME} iterations, latency {(end - start)/ITER_TIME*1000} ms, max reserved {max_reserved_memory/1024/1024/1024:8.2f} GiB, max allocated {max_allocated_memory/1024/1024/1024:8.2f} GiB")

def demo_data_parallel(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    in_shape = [20, 32]
    output_shape = [20, 5]

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    mesh = DeviceMesh("cuda", torch.arange(world_size))

    model = torch.nn.Linear(32, 5).to(device)
    nn.init.ones_(model.weight)
    nn.init.zeros_(model.bias)

    model = distribute_module(model, mesh, _data_parallel_fn, input_fn=None, output_fn=None)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, amsgrad=False)

    x = torch.randn(*in_shape).to(device).requires_grad_()
    y_grad = torch.randn(*output_shape).to(device)
    x = distribute_tensor(x, mesh, [Shard(0)])
    y_grad = distribute_tensor(y_grad, mesh, [Shard(0)])

    # warm up
    y = model(x)
    optimizer.zero_grad()
    y.backward(y_grad)
    optimizer.step()
    torch.cuda.synchronize(device)

    start = time.time()
    for i in range(ITER_TIME):
        print(f"---------------{i}--------------------")
        torch.cuda.nvtx.range_push("model_iter"+str(i))

        torch.cuda.nvtx.range_push("forward")
        y = model(x)
        torch.cuda.synchronize(device)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("zero_grad")
        optimizer.zero_grad()
        torch.cuda.synchronize(device)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("backward")
        y.backward(y_grad)
        torch.cuda.synchronize(device)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("optimizer_step")
        optimizer.step()
        torch.cuda.synchronize(device)
        torch.cuda.nvtx.range_pop()

        torch.cuda.synchronize(device)
        torch.cuda.nvtx.range_pop()
    end = time.time()
    max_reserved_memory = torch.cuda.max_memory_reserved(device)
    max_allocated_memory = torch.cuda.max_memory_allocated(device)

    print(f"rank {rank}, {ITER_TIME} iterations, latency {(end - start)/ITER_TIME*1000} ms, max reserved {max_reserved_memory/1024/1024/1024:8.2f} GiB, max allocated {max_allocated_memory/1024/1024/1024:8.2f} GiB")
    dist.destroy_process_group()

if __name__ == "__main__":
    print(f"==========Navtive Tensor 1 GPU==========")  
    native_tensor_baseline()

    print(f"==========DTensor 1 GPU==========")  
    mp.spawn(demo_data_parallel, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
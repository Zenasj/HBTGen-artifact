py
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from torch.nn.parallel import DistributedDataParallel as DDP


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Sequential(
            nn.Linear(10, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
        )
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(1024, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    model = ToyModel().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    for i in tqdm(range(1_000_000)):
        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(16, 10))
        labels = torch.randn(16, 5).to(device_id)
        loss_fn(outputs, labels).backward()
        optimizer.step()
    dist.destroy_process_group()


if __name__ == "__main__":
    demo_basic()

import torch.distributed as dist

def benchmark_with_profiler(
    benchmark_fn,
    *benchmark_fn_args,
    **benchmark_fn_kwargs,
) -> None:
    torch._C._profiler._set_cuda_sync_enabled_val(False)
    wait, warmup, active = 1, 1, 2
    num_steps = wait + warmup + active
    rank = dist.get_rank()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=1, skip_first=1
        ),
        on_trace_ready=(
            torch.profiler.tensorboard_trace_handler("./") if not rank else None
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=False,
    ) as prof:
        for step_idx in range(1, num_steps + 1):
            benchmark_fn(*benchmark_fn_args, **benchmark_fn_kwargs)
            if rank is None or rank == 0:
                prof.step()

def train_step(ddp_model, optimizer, loss_fn):
    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(16, 10))
    labels = torch.randn(16, 5).to(device_id)
    loss_fn(outputs, labels).backward()
    optimizer.step()

# Define `ddp_model`, `optimizer`, `loss_fn`
benchmark_with_profiler(train_step, ddp_model, optimizer, loss_fn)

# Init process group etc.
t = torch.empty((7347200,), device="cuda")

def fn():
    dist.all_reduce(t)

benchmark_with_profiler(fn)  # maybe change `active` to 3+ instead of 2 to get back-to-back all-reduces in the profile

# Init process group etc.
t = torch.empty((7347200,), device="cuda")

def fn():
    dist.all_reduce(t)

num_warmup, num_iters = 3, 10
for _ in range(num_warmup):
    fn()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(num_iters):
    fn()
end.record()
torch.cuda.synchronize()
elapsed_time = start.elapsed_time(end)
time_per_all_reduce = elapsed_time / num_iters
print(f"time per all-reduce: {time_per_all_reduce:.5f}")

py
import torch
import torch.distributed as dist

if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    t = torch.empty((7347200,), device=device_id)

    def fn():
        dist.all_reduce(t)

    num_warmup, num_iters = 3, 10
    for _ in range(num_warmup):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(end)
    time_per_all_reduce = elapsed_time / num_iters
    print(f"time per all-reduce {rank=}: {time_per_all_reduce:.5f}")
    dist.destroy_process_group()
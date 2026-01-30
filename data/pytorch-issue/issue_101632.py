import torch.nn as nn

import contextlib

import torch


def main(
    wait: int = 0,
    warmup: int = 0,
    active: int = 1,
    repeat: int = 2,
    skip_first: int = 0,
):
    """Main."""
    steps_per_cycle = wait + warmup + active
    # times 2 for testing extra steps.
    inner_steps = skip_first + steps_per_cycle * repeat * 2

    with _get_torch_profiler(
        wait=wait, warmup=warmup, active=active, repeat=repeat, skip_first=skip_first
    ) as p:
        for _ in range(inner_steps):
            # with _OnStep(p):
            with on_step(p):
                dummy_func()
                # p.step()


@contextlib.contextmanager
def on_step(profile):
    """On step context manager."""
    yield
    profile.step()


class _OnStep:
    def __init__(self, profile):
        self._profile = profile

    def __enter__(self):
        return None

    def __exit__(self, *args, **kwargs):
        self._profile.step()


def dummy_func():
    """A dummy function to be profiled."""
    model = torch.nn.Linear(1, 1)
    model.forward(torch.zeros(1))


def _get_schedule(wait, warmup, active, repeat, skip_first):
    return torch.profiler.schedule(
        wait=wait, warmup=warmup, active=active, repeat=repeat, skip_first=skip_first
    )


def _get_torch_profiler(wait, warmup, active, repeat, skip_first):
    activities = [
        torch.profiler.ProfilerActivity.CPU,
    ]

    return torch.profiler.profile(
        activities=activities,
        schedule=_get_schedule(
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat,
            skip_first=skip_first,
        ),
        record_shapes=True,
        with_stack=True,
        with_flops=True,
        profile_memory=False,
    )


if __name__ == "__main__":
    main()

import torch
from tqdm import tqdm
import torchvision

model = torchvision.models.resnet18().cuda()
shape = [4, 3, 2048, 2048]
iters = 65

# this runs fine.
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    with_stack=True,
    schedule=torch.profiler.schedule(
        wait=5,
        warmup=5,
        active=20,
        repeat=5),
    ) as p:
    for iter in range(iters):
        model(torch.randn(shape).cuda())
        p.step()

# this runs fine.
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    with_stack=True,
    ) as p:
    for iter in tqdm(range(iters)):
        model(torch.randn(shape).cuda())
        p.step()

# this runs fine.
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    with_stack=True,
    schedule=torch.profiler.schedule(
        wait=5,
        warmup=5,
        active=20,
        repeat=1),
    ) as p:
    for iter in tqdm(range(iters)):
        model(torch.randn(shape).cuda())
        p.step()

# this runs not fine.
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    with_stack=True,
    schedule=torch.profiler.schedule(
        wait=5,
        warmup=5,
        active=20,
        repeat=2),
    ) as p:
    for iter in tqdm(range(iters)):
        model(torch.randn(shape).cuda())
        p.step()

import torch
import torch.distributed as dist
try:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "42355"
    
    # commenting the following makes the code to work
    dist.init_process_group(rank=0, world_size=1)
    
    
    with torch.profiler.profile(with_stack=True,
        profile_memory=True,
        record_shapes=True,
        schedule=torch.profiler.schedule(
            skip_first=2,
            wait=1,
            warmup=0,
            active=5,
            repeat=1)
                                ) as p:
        for i in range(20):
            p.step()
            print(i)
            a = torch.randn(1000, 1000).to('cuda')
            b = torch.randn(1000, 1000).to('cuda')
            c = a @ b
        p.export_chrome_trace(f"profile_all_reduce.json")
        print(p.key_averages())
finally:
    if dist.is_initialized():
        dist.destroy_process_group()